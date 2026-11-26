from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F
from steerquant.backends import get_backend
from steerquant.wam_runtime import WAMFakeQuantLinear
from steerquant.kernels.w4a8 import pack_signed_int4

CUDA_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires SM80+ CUDA")


def _wam_kwargs(*, inputs: int = 256, outputs: int = 136) -> dict:
    return dict(
        qweight=torch.randint(-7, 8, (outputs, inputs), device="cuda", dtype=torch.int8),
        scales=torch.rand(outputs, 1, device="cuda") * 0.02 + 0.002,
        bias=torch.randn(outputs, device="cuda", dtype=torch.bfloat16),
        input_scale=torch.rand(inputs, device="cuda") + 0.5,
        stream_gains=torch.tensor([[2.0, 0.5], [0.5, 2.0]], device="cuda"),
        clipping_ranges=torch.tensor([3.0, 1.5], device="cuda"),
        token_fractions=torch.full((2, 2), 0.5, device="cuda"),
        weight_bits=4,
        activation_bits=4,
    )


@requires_cuda
def test_cutlass_wam_w4a4_matches_reference_and_block_fusions() -> None:
    torch.manual_seed(107)
    batch, streams, span, inputs, outputs = (1, 2, 3, 256, 136)
    kwargs = _wam_kwargs(inputs=inputs, outputs=outputs)
    reference = WAMFakeQuantLinear(**kwargs, backend=get_backend("reference"))
    module = WAMFakeQuantLinear(**kwargs, backend=get_backend("cutlass"))
    reference.set_timestep(1)
    module.set_timestep(1)
    x = torch.randn(batch, streams, span, inputs, device="cuda", dtype=torch.bfloat16)
    reference_result = reference(x)
    actual = module(x)
    gains = kwargs["stream_gains"][1].reshape(1, streams, 1, 1)
    activation_scale = kwargs["clipping_ranges"][1] / 7
    migrated = x.float() / kwargs["input_scale"]
    qactivation = torch.round(migrated * gains / activation_scale).clamp(-7, 7)
    expected = (qactivation.reshape(-1, inputs) @ kwargs["qweight"].float().T).reshape(
        batch, streams, span, outputs
    )
    expected = expected * (activation_scale / gains) * kwargs["scales"].reshape(-1) + kwargs["bias"].float()
    torch.testing.assert_close(actual.float(), expected, rtol=0.015, atol=0.03)
    torch.testing.assert_close(actual, reference_result, rtol=0.015, atol=0.2)
    modulation = torch.randn(batch, streams, 3 * inputs, device="cuda", dtype=torch.bfloat16) * 0.1
    shift, scale, _ = modulation.chunk(3, dim=-1)
    normalized = F.layer_norm(x.float(), (inputs,), eps=1e-06)
    transformed = normalized * (1 + scale.float()[:, :, None]) + shift.float()[:, :, None]
    migrated = transformed / kwargs["input_scale"]
    qactivation = torch.round(migrated * gains / activation_scale).clamp(-7, 7)
    expected_adaln = (qactivation.reshape(-1, inputs) @ kwargs["qweight"].float().T).reshape(
        batch, streams, span, outputs
    )
    expected_adaln = (
        expected_adaln * (activation_scale / gains) * kwargs["scales"].reshape(-1) + kwargs["bias"].float()
    )
    actual_adaln = module.forward_adaln(x, scale, shift, 1e-06)
    torch.testing.assert_close(actual_adaln.float(), expected_adaln, rtol=0.015, atol=0.03)
    residual = torch.randn(batch, streams * span, outputs, device="cuda", dtype=torch.bfloat16)
    gate_storage = torch.randn(batch, streams, 3 * outputs, device="cuda", dtype=torch.bfloat16)
    _, _, gate = gate_storage.chunk(3, dim=-1)
    flat_x = x.reshape(batch, streams * span, inputs)
    projected = module(flat_x)
    expanded_gate = gate[:, :, None].expand(batch, streams, span, outputs).reshape_as(projected)
    expected_residual = residual + expanded_gate * projected
    actual_residual = module.forward_gate_residual(flat_x, residual, gate)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0.015, atol=0.05)


@requires_cuda
def test_cutlass_w4a4_weight_packing_is_shared_with_w4a8() -> None:
    qweight = torch.randint(-7, 8, (16, 256), device="cuda", dtype=torch.int8)
    packed = get_backend("cutlass").pack_weight(qweight)
    torch.testing.assert_close(packed, pack_signed_int4(qweight))


from steerquant.algorithms.preprocess import randomized_hadamard_transform
from steerquant.wam_pipeline import WAMNetworkCallTracker
from steerquant.cuda_graph import CUDAGraphInferenceWrapper


@requires_cuda
def test_cutlass_preprocess_wam_w4a4_fuses_rotation_and_block_operations() -> None:
    torch.manual_seed(108)
    batch, streams, span, inputs, outputs = 1, 2, 3, 256, 136
    kwargs = _wam_kwargs(inputs=inputs, outputs=outputs)
    signs = torch.randint(0, 2, (inputs,), device="cuda", dtype=torch.float32).mul_(2).sub_(1)
    kwargs["input_rotation_signs"] = signs
    module = WAMFakeQuantLinear(**kwargs, backend=get_backend("cutlass"))
    module.set_timestep(0)
    assert module._fuses_input_rotation
    assert module.supports_fused_adaln
    assert module.supports_fused_gate_residual

    x = torch.randn(batch, streams, span, inputs, device="cuda", dtype=torch.bfloat16)
    gains = kwargs["stream_gains"][0].reshape(1, streams, 1, 1)
    activation_scale = kwargs["clipping_ranges"][0] / 7

    def integer_oracle(values: torch.Tensor) -> torch.Tensor:
        rotated = randomized_hadamard_transform(values.float(), signs)
        migrated = rotated / kwargs["input_scale"]
        qactivation = torch.round(migrated * gains / activation_scale).clamp(-7, 7)
        output = (qactivation.reshape(-1, inputs) @ kwargs["qweight"].float().T).reshape(
            batch, streams, span, outputs
        )
        return output * (activation_scale / gains) * kwargs["scales"].reshape(-1) + kwargs["bias"].float()

    actual = module(x)
    torch.testing.assert_close(actual.float(), integer_oracle(x), rtol=1.5e-2, atol=3e-2)

    modulation = torch.randn(batch, streams, 3 * inputs, device="cuda", dtype=torch.bfloat16) * 0.1
    shift, scale, _ = modulation.chunk(3, dim=-1)
    transformed = F.layer_norm(x.float(), (inputs,), eps=1e-6)
    transformed = transformed * (1 + scale.float()[:, :, None]) + shift.float()[:, :, None]
    actual_adaln = module.forward_adaln(x, scale, shift, 1e-6)
    torch.testing.assert_close(actual_adaln.float(), integer_oracle(transformed), rtol=1.5e-2, atol=3e-2)

    flat_x = x.reshape(batch, streams * span, inputs)
    residual = torch.randn(batch, streams * span, outputs, device="cuda", dtype=torch.bfloat16)
    gate_storage = torch.randn(batch, streams, 3 * outputs, device="cuda", dtype=torch.bfloat16)
    _, _, gate = gate_storage.chunk(3, dim=-1)
    projected = module(flat_x)
    expanded_gate = gate[:, :, None].expand(batch, streams, span, outputs).reshape_as(projected)
    expected_residual = residual + expanded_gate * projected
    actual_residual = module.forward_gate_residual(flat_x, residual, gate)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=1.5e-2, atol=5e-2)


@requires_cuda
def test_cutlass_wam_w4a4_cuda_graph_preserves_timestep_schedule() -> None:
    torch.manual_seed(109)
    kwargs = _wam_kwargs()
    kwargs["input_rotation_signs"] = (
        torch.randint(0, 2, (256,), device="cuda", dtype=torch.float32).mul_(2).sub_(1)
    )
    module = WAMFakeQuantLinear(**kwargs, backend=get_backend("cutlass")).eval()
    tracker = WAMNetworkCallTracker(module, num_denoise_calls=2)
    tracker.register()
    setattr(module, "_wam_network_call_tracker", tracker)
    wrapped = CUDAGraphInferenceWrapper(module, warmup=1)
    x = torch.randn(1, 2, 3, 256, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        step0 = wrapped(x).clone()
        step1 = wrapped(x).clone()
        replay0 = wrapped(x).clone()
        replay1 = wrapped(x).clone()

    assert wrapped.captured_graph_count == 2
    assert tracker.total_calls == 4
    torch.testing.assert_close(replay0, step0, rtol=0, atol=0)
    torch.testing.assert_close(replay1, step1, rtol=0, atol=0)
    assert not torch.equal(step0, step1)
