import math
import pytest
import torch
from steerquant.algorithms.preprocess import (
    randomized_hadamard_transform,
    preprocess_signs,
    apply_wam_rotations_,
    _normalized_hadamard_transform,
)


def test_rotation_orthogonality_gradient_and_local_linear():
    signs = preprocess_signs(8, seed=20260904, module_name="blocks.0.mlp.layer1")
    matrix = torch.tensor(
        [[(-1.0) ** ((i & j).bit_count()) for j in range(8)] for i in range(8)], dtype=torch.float64
    )
    rotation = signs.double()[:, None] * matrix / math.sqrt(8)
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    actual = randomized_hadamard_transform(x, signs)
    torch.testing.assert_close(actual, x @ rotation)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    torch.testing.assert_close(x.grad, grad @ rotation.T)
    weight = torch.randn(5, 8, dtype=torch.float64)
    torch.testing.assert_close(actual @ randomized_hadamard_transform(weight, signs).T, x @ weight.T)
    assert torch.equal(
        signs, preprocess_signs(8, seed=20260904, module_name="blocks.0._checkpoint_wrapped_module.mlp.layer1")
    )


def test_calibration_rotation_preserves_linear_and_captures_rotated_inputs():
    net = torch.nn.Module()
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.layer1 = torch.nn.Linear(8, 8).double()
    net.blocks = torch.nn.ModuleList([block])
    x = torch.randn(3, 8, dtype=torch.float64)
    expected = block.mlp.layer1(x)
    rotations = apply_wam_rotations_(net, 20260904)
    captured = []
    block.mlp.layer1.register_forward_pre_hook(lambda m, inputs: captured.append(inputs[0].detach()))
    actual = block.mlp.layer1(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        captured[0], randomized_hadamard_transform(x, rotations["blocks.0.mlp.layer1"])
    )
    with pytest.raises(RuntimeError, match="already installed"):
        apply_wam_rotations_(net, 20260904)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [256, 2048, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bundled_cuda_hadamard_matches_portable(width, dtype):
    torch.manual_seed(195)
    x = torch.randn(3, width, device="cuda", dtype=dtype, requires_grad=True)
    signs = preprocess_signs(width, seed=20260904, module_name="test", device="cuda")
    actual = randomized_hadamard_transform(x, signs)
    expected = _normalized_hadamard_transform(x.detach().float().cpu() * signs.cpu()).to(dtype)
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=0.008 if dtype == torch.bfloat16 else 0.002,
        atol=0.016 if dtype == torch.bfloat16 else 0.002,
    )
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
