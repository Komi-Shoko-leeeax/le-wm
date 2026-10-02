import os

os.environ["MUJOCO_GL"] = "egl"

import time
import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from codesign.quantization import configure_model_precision
from codesign.trace_io import save_trace_npz, save_metadata
from codesign.cost_adapter import PrecisionAwareShootingCostEvaluator
from stable_worldmodel.planning import GoalMSE

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        codesign_cfg = cfg.get("codesign", {})
        precision = str(codesign_cfg.get("precision", "fp16"))
        quant_scope = str(codesign_cfg.get("quant_scope", "predictor"))
        quant_state = configure_model_precision(model, precision=precision, scope=quant_scope)
        print(
            f"[codesign] precision={precision} scope={quant_scope} "
            f"weight_modules={quant_state.weight_modules} "
            f"activation_modules={quant_state.activation_modules}"
        )

        # Current stable-worldmodel LeWM exposes encode()/rollout() as a
        # Dynamics model. CEM requires a Costable object with get_cost().
        # Compose the dynamics model with the original LeWM goal-MSE objective.
        if hasattr(model, "get_cost"):
            cost_model = model
        else:
            cost_model = PrecisionAwareShootingCostEvaluator(
                model=model,
                objective=GoalMSE(),
            )
            print(
                "[codesign] wrapped Dynamics model with "
                "ShootingCostEvaluator + GoalMSE"
            )

        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(
            cfg.solver,
            model=cost_model,
            trace_enabled=bool(codesign_cfg.get("trace_enabled", True)),
            trace_candidates=bool(codesign_cfg.get("trace_candidates", True)),
            adaptive_sample=bool(codesign_cfg.get("adaptive_sample", False)),
            sample_tiers=list(codesign_cfg.get("sample_tiers", [64, 96, 128, 176, 208, 256, 300])),
            tier_thresholds=OmegaConf.to_container(codesign_cfg.get("tier_thresholds", {}), resolve=True),
            hysteresis=bool(codesign_cfg.get("hysteresis", False)),
            downshift_delay=int(codesign_cfg.get("downshift_delay", 2)),
            landscape_kwargs=OmegaConf.to_container(codesign_cfg.get("landscape", {}), resolve=True),
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    # Unified trace/metadata output for OFAT and replay analysis.
    if cfg.policy != "random" and bool(cfg.get("codesign", {}).get("trace_enabled", True)):
        trace_dir = results_path.parent / "codesign_trace"
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
            ).strip()
        except Exception:
            git_commit = "unknown"
        metadata = {
            "model": "LeWM",
            "benchmark": str(cfg.world.env_name),
            "checkpoint": str(cfg.policy),
            "git_commit": git_commit,
            "seed": int(cfg.seed),
            "precision": str(cfg.get("codesign", {}).get("precision", "fp16")),
            "quant_scope": str(cfg.get("codesign", {}).get("quant_scope", "predictor")),
            "samples": int(cfg.solver.num_samples),
            "cem_steps": int(cfg.solver.n_steps),
            "horizon": int(cfg.plan_config.horizon),
            "adaptive_sample": bool(cfg.get("codesign", {}).get("adaptive_sample", False)),
            "evaluation_time_s": float(end_time - start_time),
            "metrics": metrics,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        save_trace_npz(trace_dir / "trace.npz", policy.planning_traces, metadata)
        save_metadata(trace_dir / "metadata.json", metadata)
        (trace_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        print(f"[codesign] trace saved to {trace_dir}")


if __name__ == "__main__":
    run()
