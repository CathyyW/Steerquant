"""Utilities for matched W+A RTN experiments on the ten Linear maps per DiT block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


SLOT_NAMES = (
    "blank_control",
    "current_proprio",
    "current_wrist_image",
    "current_primary_image",
    "action",
    "future_proprio",
    "future_wrist_image",
    "future_primary_image",
    "value",
)

ROBOCASA_SLOT_NAMES = (
    "blank_control", "current_proprio", "current_wrist_image",
    "current_primary_image", "current_secondary_image", "action",
    "future_proprio", "future_wrist_image", "future_primary_image",
    "future_secondary_image", "value",
)


@dataclass(frozen=True)
class LinearSite:
    block_index: int
    linear_index: int
    linear_name: str
    module_path: str
    module: torch.nn.Linear
    slot_aligned: bool


LINEAR_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("self_q", "self_attn.q_proj", True),
    ("self_k", "self_attn.k_proj", True),
    ("self_v", "self_attn.v_proj", True),
    ("self_o", "self_attn.output_proj", True),
    ("cross_q", "cross_attn.q_proj", True),
    # Cross-attention K/V operate on text context rather than the nine WAM slots.
    ("cross_k", "cross_attn.k_proj", False),
    ("cross_v", "cross_attn.v_proj", False),
    ("cross_o", "cross_attn.output_proj", True),
    ("mlp_fc1", "mlp.layer1", True),
    ("mlp_fc2", "mlp.layer2", True),
)


def resolve_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    module = root
    for component in path.split("."):
        module = getattr(module, component)
    return module


def enumerate_block_linears(net: torch.nn.Module) -> list[LinearSite]:
    sites: list[LinearSite] = []
    if len(net.blocks) != 28:
        raise RuntimeError(f"Expected 28 DiT blocks, found {len(net.blocks)}")
    for block_index, block in enumerate(net.blocks, start=1):
        block_sites: list[LinearSite] = []
        for linear_index, (linear_name, module_path, slot_aligned) in enumerate(LINEAR_SPECS):
            module = resolve_submodule(block, module_path)
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(
                    f"block {block_index} {module_path} is {type(module).__name__}, expected nn.Linear"
                )
            block_sites.append(
                LinearSite(
                    block_index=block_index,
                    linear_index=linear_index,
                    linear_name=linear_name,
                    module_path=module_path,
                    module=module,
                    slot_aligned=slot_aligned,
                )
            )
        if len(block_sites) != 10:
            raise RuntimeError(f"Expected 10 Linear maps in block {block_index}, found {len(block_sites)}")
        sites.extend(block_sites)
    return sites


def unwrap_checkpointed_blocks(net: torch.nn.Module) -> int:
    """Replace activation-checkpoint wrappers so internal Linear outputs remain in one graph."""
    count = 0
    for index, block in enumerate(net.blocks):
        wrapped = getattr(block, "_checkpoint_wrapped_module", None)
        if wrapped is not None:
            net.blocks[index] = wrapped
            count += 1
    return count


def symmetric_rtn_per_row(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric RTN weight fake quantization."""
    if bits < 2:
        raise ValueError("weight bits must be at least 2")
    qmax = 2 ** (bits - 1) - 1
    fp32 = weight.detach().float()
    scale = fp32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
    quantized = torch.clamp(torch.round(fp32 / scale), -qmax, qmax) * scale
    return quantized.to(weight.dtype)


def symmetric_rtn_per_token(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Dynamic per-token symmetric RTN activation fake quantization."""
    if bits < 2:
        raise ValueError("activation bits must be at least 2")
    qmax = 2 ** (bits - 1) - 1
    fp32 = values.detach().float()
    scale = fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    quantized = torch.clamp(torch.round(fp32 / scale), -qmax, qmax) * scale
    return quantized.to(values.dtype)


def symmetric_rtn_per_tensor(
    values: torch.Tensor,
    bits: int,
    *,
    clipping_range: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Symmetric RTN activation fake quantization with one shared scale.

    ``clipping_range=None`` selects a dynamic range from the current tensor.
    Supplying a scalar range instead gives the static per-tensor quantizer used
    by the WAM reference runtime.
    """
    if bits < 2:
        raise ValueError("activation bits must be at least 2")
    qmax = 2 ** (bits - 1) - 1
    fp32 = values.detach().float()
    if clipping_range is None:
        clip = fp32.abs().amax().clamp_min(1e-12)
    else:
        clip = torch.as_tensor(clipping_range, device=fp32.device, dtype=torch.float32)
        if clip.numel() != 1:
            raise ValueError("per-tensor activation clipping range must be scalar")
        clip = clip.reshape(())
        if not torch.isfinite(clip) or clip <= 0:
            raise ValueError("activation clipping range must be finite and positive")
    scale = clip / qmax
    quantized = torch.clamp(torch.round(fp32 / scale), -qmax, qmax) * scale
    return quantized.to(values.dtype)


def linear_quantization_residuals(
    module: torch.nn.Linear,
    inputs: torch.Tensor,
    fp_output: torch.Tensor,
    *,
    weight_bits: int,
    activation_bits: int,
    activation_granularity: str = "per_token",
    activation_clipping_range: torch.Tensor | float | None = None,
    quantized_weight: torch.Tensor | None = None,
    reference_weight: torch.Tensor | None = None,
    variants: Iterable[str] = ("weight_only", "activation_only", "joint_wa"),
) -> dict[str, torch.Tensor]:
    """Return output residuals for W-only, A-only, and joint W+A RTN."""
    requested = tuple(dict.fromkeys(variants))
    unknown = set(requested).difference({"weight_only", "activation_only", "joint_wa"})
    if unknown:
        raise ValueError(f"Unknown quantization variants: {sorted(unknown)}")
    x = inputs.detach()
    y = fp_output.detach()
    weight = module.weight.detach() if reference_weight is None else reference_weight.detach()
    qw = symmetric_rtn_per_row(weight, weight_bits) if quantized_weight is None else quantized_weight
    if activation_granularity == "per_token":
        if activation_clipping_range is not None:
            raise ValueError("static activation clipping is only supported for per_tensor quantization")
        qx = symmetric_rtn_per_token(x, activation_bits)
    elif activation_granularity == "per_tensor":
        qx = symmetric_rtn_per_tensor(
            x,
            activation_bits,
            clipping_range=activation_clipping_range,
        )
    else:
        raise ValueError("activation_granularity must be 'per_token' or 'per_tensor'")
    bias = None if module.bias is None else module.bias.detach()
    residuals: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        if "weight_only" in requested:
            residuals["weight_only"] = F.linear(x, qw, bias) - y
        if "activation_only" in requested:
            residuals["activation_only"] = F.linear(qx, weight, bias) - y
        if "joint_wa" in requested:
            residuals["joint_wa"] = F.linear(qx, qw, bias) - y
    return residuals


def slot_grid(tensor: torch.Tensor, *, temporal: int = 9) -> torch.Tensor:
    """View a slot-aligned Linear output as [B,T,N,D]."""
    if tensor.ndim == 5:
        if tensor.shape[1] != temporal:
            raise ValueError(f"Expected temporal dimension {temporal}, got {tuple(tensor.shape)}")
        return tensor.reshape(tensor.shape[0], temporal, -1, tensor.shape[-1])
    if tensor.ndim == 3:
        if tensor.shape[1] % temporal:
            raise ValueError(f"Sequence length is not divisible by {temporal}: {tuple(tensor.shape)}")
        return tensor.reshape(tensor.shape[0], temporal, tensor.shape[1] // temporal, tensor.shape[-1])
    raise ValueError(f"Unsupported slot-aligned tensor shape: {tuple(tensor.shape)}")


class NetworkCallTracker:
    """Track the denoiser-call index used by the five-call sampler."""

    def __init__(self, net: torch.nn.Module, num_denoising_steps: int):
        self.net = net
        self.num_denoising_steps = int(num_denoising_steps)
        self.total_calls = 0
        self.current_call = -1
        self.recompute_call: int | None = None
        self.handle: Any | None = None

    def _pre_hook(self, _module: torch.nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        if self.recompute_call is not None:
            self.current_call = self.recompute_call
            return
        self.current_call = self.total_calls % self.num_denoising_steps
        self.total_calls += 1

    def register(self) -> None:
        if self.handle is not None:
            raise RuntimeError("NetworkCallTracker is already registered")
        self.handle = self.net.register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def reset(self) -> None:
        self.total_calls = 0
        self.current_call = -1

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
