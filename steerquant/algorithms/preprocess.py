"""Randomized Hadamard preprocessing for W4A4 Cosmos-Policy Linears."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn


def _canonical_module_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def preprocess_signs(
    in_features: int,
    *,
    seed: int,
    module_name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return reproducible Rademacher signs without relying on Python hashes."""

    if not is_power_of_two(in_features):
        raise ValueError(f"Preprocess requires a power-of-two input width, got {in_features}.")
    digest = hashlib.sha256(f"{int(seed)}:{_canonical_module_name(module_name)}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    signs = torch.randint(0, 2, (int(in_features),), generator=generator, dtype=torch.int8)
    signs = signs.mul_(2).sub_(1).float()
    return signs.to(device=device) if device is not None else signs


def _normalized_hadamard_transform(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 0:
        raise ValueError("Preprocess input must have a final hidden dimension.")
    width = int(values.shape[-1])
    if not is_power_of_two(width):
        raise ValueError(f"Preprocess requires a power-of-two input width, got {width}.")
    work = values
    original_shape = work.shape
    work = work.reshape(-1, width)
    butterfly = 1
    while butterfly < width:
        grouped = work.reshape(-1, width // (2 * butterfly), 2, butterfly)
        left = grouped[:, :, 0, :]
        right = grouped[:, :, 1, :]
        work = torch.cat((left + right, left - right), dim=-1).reshape(-1, width)
        butterfly *= 2
    return work.reshape(original_shape) * (1.0 / math.sqrt(width))


def _fast_normalized_hadamard_transform(values: torch.Tensor) -> torch.Tensor:
    """Run the bundled power-of-two CUDA FHT.

    The extension accepts FP16/BF16/FP32 CUDA tensors and applies the scale in
    the same kernel.  Keeping the portable butterfly below is useful for CPU
    calibration/tests, while CUDA calibration and inference use the bundled kernel.
    """

    if not values.is_cuda:
        return _normalized_hadamard_transform(values)
    width = int(values.shape[-1])
    if width < 8:
        return _normalized_hadamard_transform(values)
    from ..kernels.w4a4.loader import load_extension

    return load_extension().normalized_hadamard(values.contiguous())


class _RandomizedHadamard(torch.autograd.Function):
    """Memory-bounded transform whose backward recomputes the orthogonal inverse."""

    @staticmethod
    def forward(ctx, values: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(signs)
        with torch.no_grad():
            signed = values * signs.to(device=values.device, dtype=values.dtype)
            return _fast_normalized_hadamard_transform(signed)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (signs,) = ctx.saved_tensors
        # Forward is x D H. Its transpose/inverse is H D.
        with torch.no_grad():
            grad_input = _fast_normalized_hadamard_transform(grad_output)
            grad_input = grad_input * signs.to(device=grad_output.device, dtype=grad_output.dtype)
        return grad_input, None


def randomized_hadamard_transform(values: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply a normalized randomized Walsh-Hadamard transform on the last axis."""

    if values.ndim == 0:
        raise ValueError("Preprocess input must have a final hidden dimension.")
    width = int(values.shape[-1])
    if tuple(signs.shape) != (width,):
        raise ValueError(f"Preprocess signs must have shape [{width}], got {tuple(signs.shape)}.")
    return _RandomizedHadamard.apply(values, signs)


@torch.no_grad()
def apply_wam_rotations_(net: nn.Module, seed: int) -> dict[str, torch.Tensor]:
    """Rotate the ten WAM Linears per block before activation-cache collection."""
    from ..integrations.linear_sites import LINEAR_SPECS

    targets = {path for _, path, _ in LINEAR_SPECS}
    rotations = {}
    for raw_name, module in net.named_modules():
        name = _canonical_module_name(raw_name)
        parts = name.split(".", 2)
        if (
            not isinstance(module, nn.Linear)
            or len(parts) != 3
            or parts[0] != "blocks"
            or parts[2] not in targets
        ):
            continue
        if hasattr(module, "_preprocess_input_signs"):
            raise RuntimeError(f"Rotation already installed on {name}")
        signs = preprocess_signs(module.in_features, seed=seed, module_name=name, device=module.weight.device)
        module.weight.copy_(
            randomized_hadamard_transform(module.weight.float(), signs).to(module.weight.dtype)
        )
        module.register_buffer("_preprocess_input_signs", signs, persistent=False)

        def rotate_input(linear, inputs):
            return (randomized_hadamard_transform(inputs[0], linear._preprocess_input_signs), *inputs[1:])

        module.register_forward_pre_hook(rotate_input)
        rotations[name] = signs.cpu()
    if not rotations:
        raise ValueError("No WAM Linear sites found")
    return rotations
