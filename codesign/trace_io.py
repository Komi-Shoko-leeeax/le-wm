"""Small trace I/O helpers shared by LeWM experiment scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _convert(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _convert(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_convert(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_convert(v) for v in x)
    return x


def save_trace_npz(path: str | Path, records: list[dict], metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    converted = [_convert(r) for r in records]
    np.savez_compressed(
        path,
        records=np.asarray(converted, dtype=object),
        metadata=np.asarray([metadata or {}], dtype=object),
    )
    return path


def save_metadata(path: str | Path, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path


def load_trace_npz(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    records = data["records"].tolist()
    metadata_arr = data.get("metadata")
    metadata = metadata_arr.tolist()[0] if metadata_arr is not None else {}
    return records, metadata


def to_torch_tree(x: Any, device: str | torch.device = "cpu") -> Any:
    if isinstance(x, np.ndarray) and x.dtype != object:
        t = torch.from_numpy(x)
        return t.to(device)
    if isinstance(x, dict):
        return {k: to_torch_tree(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_torch_tree(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_torch_tree(v, device) for v in x)
    return x
