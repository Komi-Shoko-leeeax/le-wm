"""Planning-fidelity metrics for same-candidate and OFAT analysis."""
from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a, b) -> float:
    ra, rb = _rankdata(np.asarray(a)), _rankdata(np.asarray(b))
    if ra.size < 2:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    if n < 2:
        return 1.0
    ia, ja = np.triu_indices(n, 1)
    sa = np.sign(a[ia] - a[ja])
    sb = np.sign(b[ia] - b[ja])
    valid = (sa != 0) & (sb != 0)
    if not np.any(valid):
        return 1.0
    return float(np.mean(sa[valid] == sb[valid]) * 2.0 - 1.0)


def topk_overlap(a, b, k: int) -> float:
    a, b = np.asarray(a), np.asarray(b)
    k = min(int(k), len(a), len(b))
    if k <= 0:
        return 1.0
    aa = set(np.argsort(a)[:k].tolist())
    bb = set(np.argsort(b)[:k].tolist())
    return len(aa & bb) / k


def elite_overlap(ref_indices, test_indices) -> float:
    a, b = set(map(int, ref_indices)), set(map(int, test_indices))
    return len(a & b) / max(1, len(a))


def best_candidate_consistency(a, b) -> float:
    return float(int(np.argmin(a)) == int(np.argmin(b)))


def selected_action_deviation(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def energy_regret(selected_ref_energy: float, best_ref_energy: float) -> float:
    return float(selected_ref_energy - best_ref_energy)
