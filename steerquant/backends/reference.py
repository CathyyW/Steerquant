from __future__ import annotations

import torch
import torch.nn.functional as F

from ..algorithms.quantization import dequantize_weight


def output_dtype(x: torch.Tensor, keep_output_dtype: str) -> torch.dtype:
    if keep_output_dtype == "input":
        return x.dtype
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if keep_output_dtype not in mapping:
        raise ValueError(f"Unsupported keep_output_dtype: {keep_output_dtype}")
    return mapping[keep_output_dtype]


def fake_quant_activation(
    x: torch.Tensor,
    *,
    bits: int,
    symmetric: bool,
    granularity: str,
    static_scale: torch.Tensor | None = None,
    static_zero: torch.Tensor | None = None,
    group_size: int = -1,
    clip_ratio: float = 1.0,
    full_range: bool = False,
) -> torch.Tensor:
    """Reference integer fake quantization for Linear activations."""
    if not 2 <= int(bits) <= 8:
        raise ValueError(f"Activation bits must be in [2, 8], got {bits}.")
    if not 0.0 < float(clip_ratio) <= 1.0:
        raise ValueError(f"Activation clip ratio must be in (0, 1], got {clip_ratio}.")
    values = x.float()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1)) if full_range else -qmax
        if static_scale is not None:
            scale = static_scale.to(device=x.device, dtype=torch.float32).clamp_min(1e-8)
            while scale.ndim < values.ndim:
                scale = scale.unsqueeze(0)
        elif granularity == "per_tensor":
            scale = values.abs().amax().clamp_min(1e-8) * clip_ratio / qmax
        elif granularity == "per_token":
            scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) * clip_ratio / qmax
        elif granularity == "per_channel":
            scale = values.abs().reshape(-1, values.shape[-1]).amax(dim=0).clamp_min(1e-8) * clip_ratio / qmax
            while scale.ndim < values.ndim:
                scale = scale.unsqueeze(0)
        elif granularity == "per_group":
            if group_size <= 0 or values.shape[-1] % group_size:
                raise ValueError(
                    "Per-group activation quantization requires a positive group size "
                    f"dividing the last dimension; got shape={tuple(values.shape)}, group_size={group_size}."
                )
            grouped = values.reshape(-1, values.shape[-1] // group_size, group_size)
            scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) * clip_ratio / qmax
            quantized = torch.round(grouped / scale).clamp(qmin, qmax) * scale
            return quantized.reshape_as(values).to(x.dtype)
        else:
            raise ValueError(f"Unsupported activation granularity: {granularity}")
        return (torch.round(values / scale).clamp(qmin, qmax) * scale).to(x.dtype)

    qmin, qmax = 0, 2**bits - 1
    if static_scale is not None:
        if static_zero is None:
            raise ValueError("Asymmetric static activation quantization requires a zero point.")
        scale = static_scale.to(device=x.device, dtype=torch.float32).clamp_min(1e-8)
        zero = static_zero.to(device=x.device, dtype=torch.float32).clamp(qmin, qmax)
        while scale.ndim < values.ndim:
            scale = scale.unsqueeze(0)
            zero = zero.unsqueeze(0)
        return ((torch.round(values / scale + zero).clamp(qmin, qmax) - zero) * scale).to(x.dtype)
    if granularity == "per_tensor":
        lower, upper = values.amin(), values.amax()
    elif granularity == "per_token":
        lower = values.amin(dim=-1, keepdim=True)
        upper = values.amax(dim=-1, keepdim=True)
    elif granularity == "per_channel":
        flat = values.reshape(-1, values.shape[-1])
        lower, upper = flat.amin(dim=0), flat.amax(dim=0)
        while lower.ndim < values.ndim:
            lower, upper = lower.unsqueeze(0), upper.unsqueeze(0)
    elif granularity == "per_group":
        if group_size <= 0 or values.shape[-1] % group_size:
            raise ValueError(
                "Per-group activation quantization requires a positive group size "
                f"dividing the last dimension; got shape={tuple(values.shape)}, group_size={group_size}."
            )
        grouped = values.reshape(-1, values.shape[-1] // group_size, group_size)
        lower = grouped.amin(dim=-1, keepdim=True)
        upper = grouped.amax(dim=-1, keepdim=True)
        scale = (upper - lower).clamp_min(1e-5) / qmax
        zero = torch.round(-lower / scale).clamp(qmin, qmax)
        quantized = (torch.round(grouped / scale + zero).clamp(qmin, qmax) - zero) * scale
        return quantized.reshape_as(values).to(x.dtype)
    else:
        raise ValueError(f"Unsupported activation granularity: {granularity}")
    scale = ((upper - lower) / qmax).clamp_min(1e-8)
    zero = torch.round(-lower / scale).clamp(qmin, qmax)
    return ((torch.round(values / scale + zero).clamp(qmin, qmax) - zero) * scale).to(x.dtype)


class ReferenceFakeQuantBackend:
    """Portable fake-quant backend; it provides correctness, not INT speedup."""

    name = "reference"

    def linear(
        self,
        *,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        bias: torch.Tensor | None,
        group_size: int,
        symmetric: bool,
        input_scale: torch.Tensor | None,
        activation_bits: int,
        activation_granularity: str,
        activation_scale: torch.Tensor | None,
        activation_zero: torch.Tensor | None,
        fake_quant_activation_enabled: bool,
        keep_output_dtype: str,
    ) -> torch.Tensor:
        result_dtype = output_dtype(x, keep_output_dtype)
        compute_dtype = torch.float32 if result_dtype == torch.float32 else result_dtype
        x_compute = x.to(compute_dtype)
        if input_scale is not None:
            x_compute = x_compute / input_scale.to(device=x.device, dtype=compute_dtype)
        if fake_quant_activation_enabled:
            x_compute = fake_quant_activation(
                x_compute,
                bits=activation_bits,
                symmetric=symmetric,
                granularity=activation_granularity,
                static_scale=activation_scale,
                static_zero=activation_zero,
                group_size=group_size,
            )
        weight = dequantize_weight(
            qweight.to(x.device),
            scales.to(x.device),
            None if zeros is None else zeros.to(x.device),
            group_size=group_size,
            symmetric=symmetric,
            dtype=compute_dtype,
        )
        bias_compute = None if bias is None else bias.to(device=x.device, dtype=compute_dtype)
        output = F.linear(x_compute, weight, bias_compute)
        return output.to(result_dtype)
