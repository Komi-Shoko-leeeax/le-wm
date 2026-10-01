"""Build (K,U)->minimum reliable Sample Tier labels from suffix replay."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from codesign.trace_io import load_trace_npz


def _arr(x):
    return np.asarray(x)


def _baseline_trace(record):
    t = record["trace"]
    # stable-worldmodel stores batches -> iterations; JEPA stores iterations directly.
    if t and isinstance(t[0], list):
        return t[0]
    return t


def _scalar(x):
    a = np.asarray(x).reshape(-1)
    return float(a[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--planning-record", type=int, default=0)
    ap.add_argument("--action-threshold", type=float, required=True)
    ap.add_argument(
        "--energy-regret-threshold",
        type=float,
        default=None,
        help="optional maximum reference-rescored energy regret; PASS requires both thresholds",
    )
    ap.add_argument("--action-prefix", type=int, default=1, help="number of horizon actions compared")
    ap.add_argument("--output", type=Path, default=Path("tier_labels.csv"))
    args = ap.parse_args()

    base_records, _ = load_trace_npz(args.baseline)
    replay_records, _ = load_trace_npz(args.replay)
    base = base_records[args.planning_record]
    ref_action = _arr(base["selected_action"])
    # LeWM: [B,H,D]; JEPA: [H,D]. Compare only the executed prefix by default.
    if ref_action.ndim == 3:
        ref_cmp = ref_action[:, : args.action_prefix]
    else:
        ref_cmp = ref_action[: args.action_prefix]

    trace = _baseline_trace(base)
    by_step = {}
    for r in replay_records:
        if "source_iteration" not in r or "candidate_tier" not in r:
            continue
        step = int(r["source_iteration"])
        test = _arr(r["selected_action"])
        test_cmp = test[:, : args.action_prefix] if test.ndim == 3 else test[: args.action_prefix]
        dev = float(np.linalg.norm(test_cmp - ref_cmp))
        regret = _scalar(r.get("energy_regret", [np.nan]))
        by_step.setdefault(step, []).append((int(r["candidate_tier"]), dev, regret))

    rows = []
    for step, vals in sorted(by_step.items()):
        vals = sorted(vals)
        def _passes(dev, regret):
            action_ok = dev <= args.action_threshold
            if args.energy_regret_threshold is None:
                return action_ok
            return action_ok and np.isfinite(regret) and regret <= args.energy_regret_threshold
        passing = [tier for tier, dev, regret in vals if _passes(dev, regret)]
        min_tier = min(passing) if passing else max(tier for tier, _, _ in vals)
        tr = next((x for x in trace if int(x.get("step", x.get("iteration", -1))) == step), None)
        landscape = (tr or {}).get("landscape", {})
        k = int(round(_scalar(landscape.get("mode_count", [0])))) if landscape else 0
        u = _scalar(landscape.get("uncertainty", [np.nan])) if landscape else np.nan
        row = {
            "iteration": step,
            "K": k,
            "U": u,
            "minimum_reliable_tier": min_tier,
        }
        for tier, dev, regret in vals:
            row[f"action_dev_S{tier}"] = dev
            row[f"energy_regret_S{tier}"] = regret
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda x: (not x in {"iteration","K","U","minimum_reliable_tier"}, x))
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"saved {len(rows)} tier labels to {args.output}")


if __name__ == "__main__":
    main()
