"""Compare two same-candidate rescore files (e.g. FP16 vs W8A16/W8A8)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from codesign.metrics import (
    best_candidate_consistency,
    elite_overlap,
    kendall,
    spearman,
    topk_overlap,
)
from codesign.trace_io import load_trace_npz


def _flat(x):
    a = np.asarray(x)
    if a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return a.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("ranking_metrics.csv"))
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    ref, _ = load_trace_npz(args.reference)
    test, _ = load_trace_npz(args.test)
    if len(ref) != len(test):
        raise ValueError("rescore files have different iteration counts")

    rows = []
    for r, t in zip(ref, test):
        er = _flat(r["candidate_energies"])
        et = _flat(t["candidate_energies"])
        if len(er) != len(et):
            raise ValueError("candidate counts differ; same-candidate metrics are invalid")
        elite_r = _flat(r.get("elite_indices", np.argsort(er)[: args.topk])).astype(int)
        elite_t = _flat(t.get("elite_indices", np.argsort(et)[: args.topk])).astype(int)
        rows.append(
            {
                "iteration": int(r.get("source_iteration", r.get("iteration", len(rows)))),
                "num_candidates": len(er),
                "spearman": spearman(er, et),
                "kendall": kendall(er, et),
                "topk_overlap": topk_overlap(er, et, args.topk),
                "elite_overlap": elite_overlap(elite_r, elite_t),
                "best_candidate_consistency": best_candidate_consistency(er, et),
                "max_abs_energy_error": float(np.max(np.abs(er - et))),
                "mean_abs_energy_error": float(np.mean(np.abs(er - et))),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
