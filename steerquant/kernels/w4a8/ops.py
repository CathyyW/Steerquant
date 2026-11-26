from __future__ import annotations

import torch

from .loader import load_extension


def pack_signed_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Pack a symmetric ``[N, K]`` INT4 matrix along K, two values per byte."""

    if qweight.ndim != 2:
        raise ValueError(f"qweight must be two-dimensional, got {tuple(qweight.shape)}.")
    if qweight.shape[1] % 2:
        raise ValueError(f"The W4A8 kernel requires an even K dimension, got {qweight.shape[1]}.")
    if qweight.dtype != torch.int8:
        raise TypeError(f"Symmetric INT4 weights must use torch.int8 storage, got {qweight.dtype}.")
    # Quantizers already enforce this invariant. Keep the defensive range check
    # for CPU callers without introducing a device synchronization per layer.
    if not qweight.is_cuda and qweight.numel() and (qweight.min() < -8 or qweight.max() > 7):
        raise ValueError("Symmetric INT4 values must lie in [-8, 7].")
    nibbles = qweight.to(torch.uint8) & 0x0F
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()


def _validate_common(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not x.is_cuda:
        raise ValueError("The CUTLASS W4A8 kernel requires a CUDA input tensor.")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"The CUTLASS W4A8 kernel supports FP16/BF16 inputs, got {x.dtype}.")
    if x.ndim < 2:
        raise ValueError("x must have at least two dimensions.")
    if packed_weight.dtype != torch.uint8 or packed_weight.ndim != 2:
        raise TypeError("packed_weight must be a two-dimensional torch.uint8 tensor.")
    if packed_weight.device != x.device:
        raise ValueError("x and packed_weight must be on the same CUDA device.")
    k = x.shape[-1]
    n = packed_weight.shape[0]
    if packed_weight.shape[1] * 2 != k:
        raise ValueError(f"packed_weight has logical K={packed_weight.shape[1] * 2}, but x has K={k}.")
    if k % 32 or n % 8:
        raise ValueError(f"The CUTLASS W4A8 kernel requires K % 32 == 0 and N % 8 == 0; got K={k}, N={n}.")
    scales = weight_scales.float().reshape(-1).contiguous()
    if scales.numel() != n or scales.device != x.device:
        raise ValueError(f"weight_scales must contain N={n} values on the input device.")
    if bias is not None:
        if bias.device != x.device or bias.numel() != n:
            raise ValueError(f"bias must contain N={n} values on the input device.")
        bias = bias.to(dtype=x.dtype).reshape(-1).contiguous()
    if input_scale is not None:
        if input_scale.device != x.device or input_scale.numel() != k:
            raise ValueError(f"input_scale must contain K={k} values on the input device.")
        input_scale = input_scale.float().reshape(-1).contiguous()
    return x.contiguous(), scales, bias, input_scale


def symmetric_static_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    activation_scale: torch.Tensor,
    row_gains: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    input_scale_is_inverse: bool = False,
) -> torch.Tensor:
    """Static symmetric W4A8 Linear with WAM input/output row gains."""

    x_contiguous, scales, bias, input_scale = _validate_common(
        x, packed_weight, weight_scales, bias, input_scale
    )
    if activation_scale.device != x.device or activation_scale.numel() != 1:
        raise ValueError("activation_scale must be one scalar on the input device.")
    flat_rows = x.numel() // x.shape[-1]
    expanded_gains = torch.broadcast_to(row_gains, (*x.shape[:-1], 1)).reshape(-1).contiguous()
    if expanded_gains.device != x.device or expanded_gains.numel() != flat_rows:
        raise ValueError(f"row_gains must broadcast to the {flat_rows} flattened input rows.")
    expanded_gains = expanded_gains.to(dtype=x.dtype)
    return load_extension().symmetric_static(
        x_contiguous,
        packed_weight,
        scales,
        bias,
        input_scale,
        activation_scale.float().reshape(()).contiguous(),
        expanded_gains,
        bool(input_scale_is_inverse),
    )


def symmetric_static_stream_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    activation_scale: torch.Tensor,
    stream_gains: torch.Tensor,
    stream_span: int,
    bias: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    input_scale_is_inverse: bool = False,
) -> torch.Tensor:
    """Static W4A8 Linear with semantic-stream lookup fused into A8 quantization.

    Flattened rows are interpreted as repeated contiguous groups of
    ``len(stream_gains)`` streams, each containing ``stream_span`` rows.
    """

    x_contiguous, scales, bias, input_scale = _validate_common(
        x, packed_weight, weight_scales, bias, input_scale
    )
    if activation_scale.device != x.device or activation_scale.numel() != 1:
        raise ValueError("activation_scale must be one scalar on the input device.")
    gains = stream_gains.float().reshape(-1).contiguous()
    if gains.device != x.device or gains.numel() == 0:
        raise ValueError("stream_gains must be a non-empty vector on the input device.")
    stream_span = int(stream_span)
    rows = x.numel() // x.shape[-1]
    if stream_span <= 0 or rows % (stream_span * gains.numel()):
        raise ValueError("Flattened input rows must contain complete contiguous semantic-stream groups.")
    return load_extension().symmetric_static_stream(
        x_contiguous,
        packed_weight,
        scales,
        bias,
        input_scale,
        activation_scale.float().reshape(()).contiguous(),
        gains,
        stream_span,
        bool(input_scale_is_inverse),
    )
