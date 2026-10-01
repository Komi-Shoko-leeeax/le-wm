"""Offline/open-loop replay for LeWM CEM traces.

Examples:
  python -m codesign.replay_ofat --config-name pusht \
    policy=<checkpoint> codesign.replay_trace=/path/to/trace.npz \
    codesign.replay_mode=static_sample codesign.precision=fp16

Modes:
- static_sample: rerun the full saved planning case with each fixed Sample tier.
- suffix: for every saved CEM boundary, change only the next Sample tier and
  continue the suffix with reference S. Produces per-iteration tier labels.
- rescore: score the exact saved candidate banks under the selected precision.
- prefix_validate: independently rerun CEM prefixes from the same proposal/RNG.
"""

from __future__ import annotations

from pathlib import Path
import json

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm

from codesign.quantization import configure_model_precision
from codesign.trace_io import load_trace_npz, save_trace_npz, to_torch_tree


def _solver_from_record(cfg: DictConfig, model, record: dict):
    trace_batches = record["trace"]
    if not trace_batches or not trace_batches[0]:
        raise ValueError("planning record does not contain a CEM trace")
    first = trace_batches[0][0]
    mean = np.asarray(first["prev_mean"])
    batch, horizon, blocked_dim = mean.shape
    action_block = int(cfg.plan_config.action_block)
    if blocked_dim % action_block != 0:
        raise ValueError("trace action dimension is not divisible by action_block")
    base_dim = blocked_dim // action_block
    action_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(batch, base_dim),
        dtype=np.float32,
    )
    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    c = cfg.get("codesign", {})
    solver = hydra.utils.instantiate(
        cfg.solver,
        model=model,
        trace_enabled=True,
        trace_candidates=True,
        adaptive_sample=False,
        sample_tiers=list(c.get("sample_tiers", [64, 96, 128, 176, 208, 256, 300])),
        tier_thresholds=OmegaConf.to_container(c.get("tier_thresholds", {}), resolve=True),
        hysteresis=False,
        landscape_kwargs=OmegaConf.to_container(c.get("landscape", {}), resolve=True),
    )
    solver.configure(action_space=action_space, n_envs=batch, config=plan_cfg)
    return solver


@hydra.main(version_base=None, config_path="../config/eval", config_name="pusht")
def run(cfg: DictConfig):
    c = cfg.get("codesign", {})
    trace_path = c.get("replay_trace")
    if not trace_path:
        raise ValueError("set codesign.replay_trace=/path/to/trace.npz")
    if cfg.policy == "random":
        raise ValueError("replay requires a model checkpoint in policy=...")

    records, metadata = load_trace_npz(trace_path)
    call_idx = int(c.get("replay_planning_call", 0))
    if call_idx >= len(records):
        raise IndexError(f"replay_planning_call={call_idx}, only {len(records)} records")
    record = records[call_idx]

    model = swm.wm.utils.load_pretrained(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    precision = str(c.get("precision", "fp16"))
    quant_scope = str(c.get("quant_scope", "predictor"))
    qstate = configure_model_precision(model, precision=precision, scope=quant_scope)
    print(
        f"[codesign replay] precision={precision} scope={quant_scope} "
        f"W-modules={qstate.weight_modules} A-modules={qstate.activation_modules}"
    )

    solver = _solver_from_record(cfg, model, record)
    info = to_torch_tree(record["replay_info_dict"], device="cpu")
    trace = to_torch_tree(record["trace"][0], device="cpu")
    # Reference energy of the baseline final mean under the currently loaded model.
    # For tier calibration this runner should be invoked with precision=fp16.
    baseline_action = to_torch_tree(record["selected_action"], device=solver.device)
    baseline_ref_energy = solver.score_candidates(info, baseline_action.unsqueeze(1)).detach().cpu()
    tiers = [int(x) for x in c.get("sample_tiers", [64, 96, 128, 176, 208, 256, 300])]
    mode = str(c.get("replay_mode", "suffix"))
    out_records = []

    if mode == "static_sample":
        for tier in tiers:
            out = solver.replay_static_sample(info, trace, tier)
            selected = out["actions"].to(solver.device)
            rescored = solver.score_candidates(info, selected.unsqueeze(1)).detach().cpu()
            out_records.append(
                {
                    "candidate_tier": tier,
                    "selected_action": out["actions"],
                    "reference_rescored_energy": rescored,
                    "baseline_reference_energy": baseline_ref_energy,
                    "energy_regret": rescored - baseline_ref_energy,
                    "final_costs": out["costs"],
                    "sample_sequence": out["sample_sequence"],
                    "total_candidate_evaluations": out["total_candidate_evaluations"],
                    "trace": out.get("trace"),
                }
            )

    elif mode == "suffix":
        for step in trace[:-1]:
            for tier in tiers:
                out = solver.replay_suffix(
                    info,
                    step,
                    tier,
                    reference_num_samples=int(cfg.solver.num_samples),
                )
                selected = out["actions"].to(solver.device)
                rescored = solver.score_candidates(info, selected.unsqueeze(1)).detach().cpu()
                out_records.append(
                    {
                        "source_iteration": int(step["step"]),
                        "candidate_tier": tier,
                        "selected_action": out["actions"],
                        "reference_rescored_energy": rescored,
                        "baseline_reference_energy": baseline_ref_energy,
                        "energy_regret": rescored - baseline_ref_energy,
                        "final_costs": out["costs"],
                        "sample_sequence": out["sample_sequence"],
                        "total_candidate_evaluations": out["total_candidate_evaluations"],
                        "trace": out.get("trace"),
                    }
                )

    elif mode == "rescore":
        for step in trace:
            if "candidates" not in step:
                raise ValueError("trace_candidates must be true for same-candidate rescore")
            candidates = to_torch_tree(step["candidates"], device=solver.device)
            energies = solver.score_candidates(info, candidates).detach().cpu()
            out_records.append(
                {
                    "source_iteration": int(step["step"]),
                    "num_samples": int(step["num_samples"]),
                    "precision": precision,
                    "candidate_energies": energies,
                    "elite_indices": torch.argsort(energies, dim=1)[:, : min(int(cfg.solver.topk), energies.shape[1])],
                }
            )

    elif mode == "prefix_validate":
        for prefix_steps in range(1, len(trace) + 1):
            out = solver.replay_prefix(info, trace, prefix_steps)
            expected = to_torch_tree(trace[prefix_steps - 1]["mean"], device="cpu")
            actual = out["actions"]
            max_abs = float((actual - expected).abs().max().item())
            out_records.append(
                {
                    "prefix_steps": prefix_steps,
                    "max_abs_action_diff": max_abs,
                    "selected_action": actual,
                }
            )
    else:
        raise ValueError("codesign.replay_mode must be static_sample, suffix, rescore, or prefix_validate")

    output = Path(str(c.get("replay_output", "codesign_replay.npz")))
    replay_meta = {
        **metadata,
        "replay_mode": mode,
        "replay_precision": precision,
        "replay_quant_scope": quant_scope,
        "planning_call": call_idx,
        "tiers": tiers,
    }
    save_trace_npz(output, out_records, replay_meta)
    output.with_suffix(".json").write_text(json.dumps(replay_meta, indent=2, default=str), encoding="utf-8")
    print(f"[codesign replay] saved {len(out_records)} records to {output}")


if __name__ == "__main__":
    run()
