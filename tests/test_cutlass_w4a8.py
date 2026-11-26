from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F
from steerquant.backends import get_backend
from steerquant.wam_runtime import WAMFakeQuantLinear

CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires SM80+ CUDA")


@requires_cuda
def test_wam_cutlass_matches_reference_backend() -> None:
    torch.manual_seed(23)
    inputs, outputs = (64, 72)
    kwargs = dict(
        qweight=torch.randint(-7, 8, (outputs, inputs), device="cuda", dtype=torch.int8),
        scales=torch.rand(outputs, 1, device="cuda") * 0.02 + 0.002,
        bias=torch.randn(outputs, device="cuda", dtype=torch.bfloat16),
        input_scale=torch.rand(inputs, device="cuda") + 0.5,
        stream_gains=torch.tensor([[2.0, 0.5]], device="cuda"),
        clipping_ranges=torch.tensor([3.0], device="cuda"),
        token_fractions=torch.tensor([[0.5, 0.5]], device="cuda"),
        weight_bits=4,
        activation_bits=8,
    )
    reference = WAMFakeQuantLinear(**kwargs, backend=get_backend("reference"))
    cutlass = WAMFakeQuantLinear(**kwargs, backend=get_backend("cutlass"))
    reference.set_timestep(0)
    cutlass.set_timestep(0)
    x = torch.randn(1, 6, inputs, device="cuda", dtype=torch.bfloat16)
    expected = reference(x)
    actual = cutlass(x)
    assert cutlass.qweight.shape == (outputs, inputs // 2)
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.08)


@requires_cuda
def test_wam_cutlass_block_fusions_match_unfused_wam() -> None:
    torch.manual_seed(37)
    batch, streams, stream_span, inputs, outputs = (1, 2, 3, 64, 72)
    module = WAMFakeQuantLinear(
        qweight=torch.randint(-7, 8, (outputs, inputs), device="cuda", dtype=torch.int8),
        scales=torch.rand(outputs, 1, device="cuda") * 0.02 + 0.002,
        bias=None,
        input_scale=torch.rand(inputs, device="cuda") + 0.5,
        stream_gains=torch.tensor([[2.0, 0.5]], device="cuda"),
        clipping_ranges=torch.tensor([4.0], device="cuda"),
        token_fractions=torch.tensor([[0.5, 0.5]], device="cuda"),
        weight_bits=4,
        activation_bits=8,
        backend=get_backend("cutlass"),
    )
    module.set_timestep(0)
    x = torch.randn(batch, streams, stream_span, 1, inputs, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(batch, streams, 3 * inputs, device="cuda", dtype=torch.bfloat16) * 0.1
    shift, scale, _ = modulation.chunk(3, dim=-1)
    broadcast_scale = scale[:, :, None, None]
    broadcast_shift = shift[:, :, None, None]
    normalized = F.layer_norm(x, (inputs,), eps=1e-06) * (1 + broadcast_scale) + broadcast_shift
    expected_adaln = module(normalized)
    actual_adaln = module.forward_adaln(x, scale, shift, 1e-06)
    torch.testing.assert_close(actual_adaln, expected_adaln, rtol=0.015, atol=0.09)
    hidden = torch.randn(batch, streams * stream_span, inputs, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(batch, streams * stream_span, outputs, device="cuda", dtype=torch.bfloat16)
    gate_storage = torch.randn(batch, streams, 3 * outputs, device="cuda", dtype=torch.bfloat16)
    _, _, gate = gate_storage.chunk(3, dim=-1)
    projected = module(hidden)
    expanded_gate = gate[:, :, None, :].expand(batch, streams, stream_span, outputs).reshape_as(projected)
    expected_residual = residual + expanded_gate * projected
    actual_residual = module.forward_gate_residual(hidden, residual, gate)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0.008, atol=0.04)
