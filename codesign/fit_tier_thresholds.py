"""Fit conservative DirectTierPolicy U-thresholds from calibrated tier labels.

The fit is intentionally simple and hardware-friendly. For each active-mode count K
and each tier boundary, it chooses a low quantile of U among states that require a
higher tier. ``--underalloc-quantile 0`` is fully conservative on the calibration
set; small positive values trade a controlled amount of under-allocation for lower
sample cost and should be validated on held-out data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--tiers", type=int, nargs="+", default=[64, 96, 128, 176, 208, 256, 300])
    ap.add_argument("--underalloc-quantile", type=float, default=0.0)
    ap.add_argument("--output", type=Path, default=Path("tier_thresholds.json"))
    args = ap.parse_args()
    tiers = sorted(args.tiers)
    q = float(args.underalloc_quantile)
    if not 0 <= q < 1:
        raise ValueError("underalloc-quantile must be in [0,1)")

    rows = []
    with args.labels.open() as f:
        for r in csv.DictReader(f):
            rows.append((int(r["K"]), float(r["U"]), int(r["minimum_reliable_tier"])))
    by_k = {}
    for k, u, t in rows:
        if np.isfinite(u):
            by_k.setdefault(k, []).append((u, t))

    thresholds = {}
    for k, vals in sorted(by_k.items()):
        cuts = []
        for tier in tiers[:-1]:
            harder_u = np.array([u for u, req in vals if req > tier], dtype=float)
            if harder_u.size == 0:
                cut = 1.000001
            else:
                cut = float(np.quantile(harder_u, q))
            cuts.append(cut)
        # DirectTierPolicy expects ascending thresholds.
        cuts = np.maximum.accumulate(np.asarray(cuts)).clip(0.0, 1.000001).tolist()
        thresholds[str(k)] = cuts

    payload = {
        "tiers": tiers,
        "underalloc_quantile": q,
        "tier_thresholds": thresholds,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
