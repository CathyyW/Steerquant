"""Action-probe sensitivity collector for the matched W+A reference quantizer."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any
import torch
from .algorithms.preprocess import preprocess_signs, randomized_hadamard_transform
from .integrations.linear_sites import (
    SLOT_NAMES,
    LinearSite,
    NetworkCallTracker,
    enumerate_block_linears,
    linear_quantization_residuals,
    slot_grid,
    symmetric_rtn_per_row,
)


@dataclass
class LinearCapture:
    site: LinearSite
    denoise_call: int
    inputs: torch.Tensor | None
    output: torch.Tensor | None
    rows: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict)
    probe_sq_sum: dict[int, float] = field(default_factory=dict)


class FirstOrderLinearCollector:
    def __init__(
        self,
        net: torch.nn.Module,
        *,
        weight_bits: int,
        activation_bits: int,
        activation_granularity: str,
        activation_clipping_ranges: dict[str, torch.Tensor] | None,
        num_action_probes: int,
        num_action_elements: int,
        num_denoising_steps: int,
        rotation_seed: int | None = None,
        slot_names: tuple[str, ...] = SLOT_NAMES,
        checkpoint_denoisers: bool = False,
        reuse_forward: bool = False,
    ) -> None:
        self.net = net
        self.checkpoint_denoisers = checkpoint_denoisers
        self.reuse_forward = reuse_forward
        self.capture_mode = "normal"
        self.capture_lookup: dict[tuple[int, int], LinearCapture] = {}
        self.backward_seen: set[tuple[int, int]] = set()
        self.slot_names = tuple(slot_names)
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.activation_granularity = str(activation_granularity)
        self.activation_clipping_ranges = activation_clipping_ranges
        self.num_action_probes = int(num_action_probes)
        self.num_action_elements = int(num_action_elements)
        self.tracker = NetworkCallTracker(net, num_denoising_steps)
        self.sites = enumerate_block_linears(net)
        self.rotation_signs = (
            {
                id(site.module): preprocess_signs(
                    site.module.in_features,
                    seed=rotation_seed,
                    module_name=self._module_name(site),
                    device=site.module.weight.device,
                )
                for site in self.sites
            }
            if rotation_seed is not None
            else {}
        )
        self.handles: list[Any] = []
        self.captures: list[LinearCapture] = []
        self.quantized_weights: dict[int, torch.Tensor] = {}
        self.backward_mode = "idle"
        self.probe_index = -1
        self.probe_sq_totals: dict[tuple[int, int, int, int], float] = {}

    @staticmethod
    def _module_name(site: LinearSite) -> str:
        return f"blocks.{site.block_index - 1}.{site.module_path}"

    def _activation_clipping_range(self, capture: LinearCapture) -> torch.Tensor | None:
        if self.activation_clipping_ranges is None:
            return None
        module_name = self._module_name(capture.site)
        try:
            ranges = self.activation_clipping_ranges[module_name]
        except KeyError as error:
            raise KeyError(f"Missing static activation range for {module_name}") from error
        if not 0 <= capture.denoise_call < ranges.numel():
            raise IndexError(
                f"Denoise call {capture.denoise_call} is outside the static ranges for {module_name}"
            )
        return ranges[capture.denoise_call]

    @property
    def _network_calls_seen(self) -> int:
        """Compatibility with the shared differentiable sampler."""
        return self.tracker.total_calls

    def begin_denoise_call(self, call_index: int, _sigma: torch.Tensor) -> None:
        """Expose the call index before the denoiser enters the network."""
        self.tracker.current_call = int(call_index)

    def checkpoint_call(self, function, current, sigma, call_index):
        """Keep only solver boundaries; reconstruct one denoiser for backward.

        Reentrant checkpointing runs the first pass without gradients. Its
        recomputation installs the Linear gradient hooks exactly once, with
        the original call index and without advancing the network counter.
        """
        from torch.utils.checkpoint import checkpoint

        def run(x, s):
            previous_mode = self.capture_mode
            previous_call = self.tracker.recompute_call
            recomputing = torch.is_grad_enabled()
            self.capture_mode = "recompute" if recomputing else "metadata"
            self.tracker.recompute_call = call_index if recomputing else None
            try:
                return function(x, s)
            finally:
                self.capture_mode = previous_mode
                self.tracker.recompute_call = previous_call

        return checkpoint(run, current, sigma, use_reentrant=True, preserve_rng_state=True)

    def _weight(self, site: LinearSite) -> torch.Tensor:
        key = id(site.module)
        if key not in self.quantized_weights:
            signs = self.rotation_signs.get(key)
            self.quantized_weights[key] = (
                symmetric_rtn_per_row(site.module.weight, self.weight_bits)
                if signs is None
                else self._quantize_rotated_weight(site.module.weight, signs)
            )
        return self.quantized_weights[key]

    @torch.no_grad()
    def _quantize_rotated_weight(
        self,
        weight: torch.Tensor,
        signs: torch.Tensor,
        *,
        output_row_chunk: int = 128,
    ) -> torch.Tensor:
        """Rotate and W-quantize row chunks without a full fp32 weight copy."""
        qmax = 2 ** (self.weight_bits - 1) - 1
        quantized = torch.empty_like(weight)
        for start in range(0, weight.shape[0], output_row_chunk):
            end = min(start + output_row_chunk, weight.shape[0])
            rotated = randomized_hadamard_transform(weight[start:end].detach().float(), signs)
            scale = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
            block = torch.clamp(torch.round(rotated / scale), -qmax, qmax) * scale
            quantized[start:end].copy_(block.to(weight.dtype))
        return quantized

    def _inputs(self, capture: LinearCapture) -> torch.Tensor:
        if capture.inputs is None:
            raise RuntimeError("Capture inputs have already been released")
        values = capture.inputs.to(capture.site.module.weight.device)
        signs = self.rotation_signs.get(id(capture.site.module))
        return values if signs is None else randomized_hadamard_transform(values, signs)

    def _parts(self, tensor: torch.Tensor, slot_aligned: bool) -> list[tuple[int, torch.Tensor]]:
        if not slot_aligned:
            return [(-1, tensor)]
        grid = slot_grid(tensor, temporal=len(self.slot_names))
        return [(slot_index, grid[:, slot_index]) for slot_index in range(len(self.slot_names))]

    def _backward_hook(self, capture: LinearCapture, grad: torch.Tensor) -> torch.Tensor:
        if self.backward_mode == "action_probe":
            key = (id(capture.site.module), capture.denoise_call)
            if key in self.backward_seen:
                raise RuntimeError("Duplicate sensitivity backward hook")
            self.backward_seen.add(key)
            if capture.output is None:
                raise RuntimeError("Capture output has already been released")
            residual = linear_quantization_residuals(
                capture.site.module,
                self._inputs(capture),
                capture.output.to(grad.device),
                weight_bits=self.weight_bits,
                activation_bits=self.activation_bits,
                activation_granularity=self.activation_granularity,
                activation_clipping_range=self._activation_clipping_range(capture),
                quantized_weight=self._weight(capture.site),
                variants=("joint_wa",),
            )["joint_wa"]
            grad_parts = dict(self._parts(grad.detach(), capture.site.slot_aligned))
            for slot_index, values in self._parts(residual, capture.site.slot_aligned):
                directional = float((grad_parts[slot_index].float() * values.float()).sum().item())
                self._ensure_finite(directional, capture, "joint_wa", slot_index)
                key = (capture.site.block_index, capture.site.linear_index, capture.denoise_call, slot_index)
                self.probe_sq_totals[key] = self.probe_sq_totals.get(key, 0.0) + directional**2
            if self._release_probe_captures:
                capture.inputs = None
                capture.output = None
        return grad

    @staticmethod
    def _ensure_finite(value: float, capture: LinearCapture, variant: str, slot_index: int) -> None:
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Non-finite score at block={capture.site.block_index}, linear={capture.site.linear_name}, call={capture.denoise_call}, variant={variant}, slot={slot_index}"
            )

    def _forward_hook(
        self, site: LinearSite, _module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        key = (id(site.module), self.tracker.current_call)
        if self.capture_mode == "metadata":
            if key in self.capture_lookup:
                raise RuntimeError("Duplicate sensitivity forward capture")
            capture = LinearCapture(site, self.tracker.current_call, None, None)
            self.captures.append(capture)
            self.capture_lookup[key] = capture
            return output
        if not tensor.requires_grad:
            if site.slot_aligned:
                raise RuntimeError(f"{site.linear_name} output does not require gradients")
            tensor.requires_grad_(True)
        if self.capture_mode == "recompute":
            capture = self.capture_lookup[key]
            # Share the recomputation's GPU storage; do not create CPU copies.
            capture.inputs = inputs[0].detach()
            capture.output = tensor.detach()
        else:
            capture = LinearCapture(
                site=site,
                denoise_call=self.tracker.current_call,
                inputs=inputs[0].detach().cpu(),
                output=tensor.detach().cpu(),
            )
            self.captures.append(capture)
        tensor.register_hook(lambda grad, item=capture: self._backward_hook(item, grad))
        return output

    def register(self) -> None:
        self.tracker.register()
        for site in self.sites:
            self.handles.append(
                site.module.register_forward_hook(
                    lambda module, inputs, output, selected=site: self._forward_hook(
                        selected, module, inputs, output
                    )
                )
            )

    def reset_observation(self) -> None:
        self.probe_sq_totals.clear()
        self.quantized_weights.clear()
        self.begin_pass()

    def begin_pass(self) -> None:
        self.captures.clear()
        self.capture_lookup.clear()
        self.backward_seen.clear()
        self.tracker.reset()
        self.backward_mode = "idle"
        self.probe_index = -1

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.tracker.remove()
        self.quantized_weights.clear()
        self.captures.clear()
        self.capture_lookup.clear()

    def set_probe(self, index: int) -> None:
        if not 0 <= index < self.num_action_probes:
            raise ValueError("Probe index is outside the configured projection count")
        if self.reuse_forward and index != self.probe_index + 1:
            raise ValueError("Retained-graph probes must run once each, in order")
        self.backward_seen.clear()
        self.backward_mode = "action_probe"
        self.probe_index = int(index)

    @property
    def _release_probe_captures(self) -> bool:
        return (not self.reuse_forward or self.checkpoint_denoisers
                or self.probe_index + 1 == self.num_action_probes)

    def validate_capture_count(self) -> None:
        expected = len(self.sites) * self.tracker.num_denoising_steps
        if len(self.captures) != expected:
            raise RuntimeError(f"Expected {expected} Linear captures, observed {len(self.captures)}")
        if self.tracker.total_calls != self.tracker.num_denoising_steps:
            raise RuntimeError(
                f"Expected {self.tracker.num_denoising_steps} network calls, observed {self.tracker.total_calls}"
            )

    def validate_backward_count(self) -> None:
        if len(self.backward_seen) != len(self.captures):
            raise RuntimeError(f"Missing sensitivity gradients: {len(self.backward_seen)}/{len(self.captures)}")
        if self._release_probe_captures and any(c.inputs is not None or c.output is not None for c in self.captures):
            raise RuntimeError("Sensitivity activations were not released after backward")
        if not self._release_probe_captures and any(c.inputs is None or c.output is None for c in self.captures):
            raise RuntimeError("Sensitivity activations were released before the final retained-graph probe")

    def export_action_probe_rows(self, base: dict[str, Any]) -> list[dict[str, Any]]:
        """Export the joint W+A action score without auxiliary GT-loss metrics."""
        rows: list[dict[str, Any]] = []
        for capture in self.captures:
            expected_slots = range(len(self.slot_names)) if capture.site.slot_aligned else (-1,)
            for slot_index in expected_slots:
                probe_key = (
                    capture.site.block_index,
                    capture.site.linear_index,
                    capture.denoise_call,
                    slot_index,
                )
                probe_sq = self.probe_sq_totals.get(probe_key, math.nan)
                action_rmse = math.sqrt(probe_sq / self.num_action_probes / self.num_action_elements)
                self._ensure_finite(action_rmse, capture, "joint_wa", slot_index)
                rows.append(
                    {
                        **base,
                        "block_index": capture.site.block_index,
                        "linear_index": capture.site.linear_index,
                        "linear_name": capture.site.linear_name,
                        "module_path": capture.site.module_path,
                        "denoise_call": capture.denoise_call,
                        "slot_aligned": capture.site.slot_aligned,
                        "slot_index": slot_index,
                        "slot_name": self.slot_names[slot_index] if slot_index >= 0 else "global_text_context",
                        "quantization_variant": "joint_wa",
                        "predicted_standardized_action_rmse": action_rmse,
                    }
                )
        return rows
