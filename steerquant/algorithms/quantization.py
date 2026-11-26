from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class QuantizedLinearWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor | None
    input_scale: torch.Tensor | None = None
    activation_scale: torch.Tensor | None = None
    activation_zero: torch.Tensor | None = None
    input_rotation_signs: torch.Tensor | None = None


def effective_group_size(group_size: int, in_features: int) -> int:
    if group_size <= 0 or group_size > in_features:
        return in_features
    return int(group_size)


def quantize_weight_rtn(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    granularity: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize a 2D Linear weight with reference round-to-nearest PTQ."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D Linear weight, got {tuple(weight.shape)}.")
    if not 2 <= int(bits) <= 8:
        raise ValueError(f"`bits` must be in [2, 8], got {bits}.")

    out_features, in_features = weight.shape
    per_tensor = granularity == "per_tensor"
    if granularity in ("per_tensor", "per_channel"):
        group_size = in_features
    elif granularity == "per_group":
        group_size = effective_group_size(group_size, in_features)
    else:
        raise ValueError(f"Unsupported weight granularity: {granularity}")
    num_groups = 1 if per_tensor else math.ceil(in_features / group_size)
    scale_rows = 1 if per_tensor else out_features

    q_dtype = torch.int8 if symmetric else torch.uint8
    qweight = torch.empty_like(weight, dtype=q_dtype)
    scales = torch.empty((scale_rows, num_groups), dtype=torch.float32, device=weight.device)
    zeros = (
        None if symmetric else torch.empty((scale_rows, num_groups), dtype=torch.int16, device=weight.device)
    )
    weight_f = weight.detach().float()

    for group_index in range(num_groups):
        start = group_index * group_size
        end = min(start + group_size, in_features)
        block = weight_f[:, start:end]
        if symmetric:
            qmax = 2 ** (bits - 1) - 1
            qmin = -qmax
            if per_tensor:
                scale = block.abs().amax().clamp_min(1e-8) / qmax
                q = torch.round(block / scale).clamp(qmin, qmax)
            else:
                scale = block.abs().amax(dim=1).clamp_min(1e-8) / qmax
                q = torch.round(block / scale[:, None]).clamp(qmin, qmax)
            scales[:, group_index] = scale
        else:
            qmin, qmax = 0, 2**bits - 1
            if per_tensor:
                lower, upper = block.amin(), block.amax()
            else:
                lower, upper = block.amin(dim=1), block.amax(dim=1)
            scale = ((upper - lower) / qmax).clamp_min(1e-8)
            zero = torch.round(-lower / scale).clamp(qmin, qmax)
            if per_tensor:
                q = torch.round(block / scale + zero).clamp(qmin, qmax)
            else:
                q = torch.round(block / scale[:, None] + zero[:, None]).clamp(qmin, qmax)
            scales[:, group_index] = scale
            assert zeros is not None
            zeros[:, group_index] = zero.to(torch.int16)
        qweight[:, start:end] = q.to(q_dtype)
    return qweight, scales, zeros


def dequantize_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    *,
    group_size: int,
    symmetric: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    if qweight.ndim != 2:
        raise ValueError(f"Expected a 2D qweight, got {tuple(qweight.shape)}.")
    out_features, in_features = qweight.shape
    per_tensor = scales.shape[0] == 1
    if scales.shape[1] == 1:
        group_size = in_features
    else:
        group_size = effective_group_size(group_size, in_features)
    num_groups = math.ceil(in_features / group_size)
    if scales.shape[1] != num_groups:
        raise ValueError(f"Expected {num_groups} scale groups, got {scales.shape[1]}.")
    if not per_tensor and scales.shape[0] != out_features:
        raise ValueError(f"Expected 1 or {out_features} scale rows, got {scales.shape[0]}.")
    if not symmetric and zeros is None:
        raise ValueError("Asymmetric dequantization requires zero points.")

    weight = torch.empty((out_features, in_features), dtype=dtype, device=qweight.device)
    qweight_f = qweight.float()
    for group_index in range(num_groups):
        start = group_index * group_size
        end = min(start + group_size, in_features)
        scale = scales[:, group_index].float()
        zero = None if zeros is None else zeros[:, group_index].float()
        if not per_tensor:
            scale = scale[:, None]
            zero = None if zero is None else zero[:, None]
        block = qweight_f[:, start:end] * scale if symmetric else (qweight_f[:, start:end] - zero) * scale
        weight[:, start:end] = block.to(dtype)
    return weight
