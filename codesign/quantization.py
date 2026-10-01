"""Algorithmic precision/quantization controls for planning experiments.

W8A16/W8A8 here are fidelity experiments, not claims of native CUDA INT8
speedup. Weights are symmetrically quantized then dequantized once; W8A8 also
fake-quantizes activations at selected Linear/Conv boundaries. Compute runs
under CUDA FP16 autocast for fp16/w8a16/w8a8.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


PRECISIONS = {"fp16", "w8a16", "w8a8", "fp32"}
QUANT_SCOPES = {"predictor", "encoder", "all"}


def fake_quant_tensor(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    if not torch.is_tensor(x) or not x.is_floating_point() or x.numel() == 0:
        return x
    qmax = (1 << (bits - 1)) - 1
    amax = x.detach().abs().amax()
    if not torch.isfinite(amax) or float(amax) == 0.0:
        return x
    scale = amax / qmax
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def _fake_quant_weight_per_out_channel(weight: torch.Tensor, bits: int = 8) -> torch.Tensor:
    if weight.ndim < 2:
        return fake_quant_tensor(weight, bits)
    qmax = (1 << (bits - 1)) - 1
    reduce_dims = tuple(range(1, weight.ndim))
    amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = (amax / qmax).clamp_min(torch.finfo(weight.dtype).eps)
    q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    return q * scale


def _map_tensors(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_tensors(x, fn) for x in obj)
    if isinstance(obj, list):
        return [_map_tensors(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    return obj


def _scope_modules(model: nn.Module, scope: str) -> list[nn.Module]:
    if scope not in QUANT_SCOPES:
        raise ValueError(f"quant_scope must be one of {sorted(QUANT_SCOPES)}, got {scope}")
    if scope == "all":
        return [model]
    if not hasattr(model, scope):
        raise AttributeError(f"model has no module named {scope!r}")
    return [getattr(model, scope)]


@dataclass
class QuantizationState:
    precision: str
    scope: str
    weight_modules: int
    activation_modules: int
    hook_handles: list

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


def configure_model_precision(
    model: nn.Module,
    precision: str = "fp16",
    scope: str = "predictor",
) -> QuantizationState:
    precision = precision.lower()
    scope = scope.lower()
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision}")

    # Expose execution mode to the JEPA get_cost path.
    model.codesign_precision = precision
    model.codesign_quant_scope = scope

    if precision in {"fp16", "fp32"}:
        return QuantizationState(precision, scope, 0, 0, [])

    target_roots = _scope_modules(model, scope)
    compute_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    seen = set()
    weight_count = 0
    activation_count = 0
    handles = []

    for root in target_roots:
        for module in root.modules():
            if id(module) in seen or not isinstance(module, compute_types):
                continue
            seen.add(id(module))
            if getattr(module, "weight", None) is not None:
                with torch.no_grad():
                    module.weight.copy_(_fake_quant_weight_per_out_channel(module.weight))
                weight_count += 1

            if precision == "w8a8":
                def pre_hook(_module, args):
                    return _map_tensors(args, fake_quant_tensor)

                def post_hook(_module, _args, output):
                    return _map_tensors(output, fake_quant_tensor)

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook))
                activation_count += 1

    return QuantizationState(
        precision=precision,
        scope=scope,
        weight_modules=weight_count,
        activation_modules=activation_count,
        hook_handles=handles,
    )


def autocast_context(module: nn.Module):
    precision = getattr(module, "codesign_precision", "fp32")
    use_fp16 = precision in {"fp16", "w8a16", "w8a8"}
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if use_fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()
