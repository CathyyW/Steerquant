"""Reference inference runtime for calibrated WAM Linear modules.

The runtime expands stream-level gains only to broadcastable token rows; it
never materializes the token-level diagonal matrix from the WAM notation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
import torch.nn as nn

from .algorithms.preprocess import randomized_hadamard_transform
from .algorithms.wam_d import WAMDCalibrationResult
from .algorithms.wam_gamma import WAMGammaCalibrationResult
from .backends import get_backend
from .backends.reference import fake_quant_activation


class _WAMTimestepState:
    """One Python schedule value shared by every WAM Linear in a DiT."""

    def __init__(self, num_denoise_calls: int, module_count: int) -> None:
        self.num_denoise_calls = int(num_denoise_calls)
        self.module_count = int(module_count)
        self.value: int | None = None

    def set(self, timestep: int) -> None:
        timestep = int(timestep)
        if not 0 <= timestep < self.num_denoise_calls:
            raise IndexError(f"timestep {timestep} is outside [0, {self.num_denoise_calls}).")
        self.value = timestep


def expand_wam_stream_gains(
    x: torch.Tensor,
    stream_gains: torch.Tensor,
    *,
    stream_axis: int | None = None,
    stream_token_counts: Sequence[int] | None = None,
    stream_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand ``[num_streams]`` gains to a tensor broadcastable over ``x``.

    Exactly one optional layout description may be supplied:

    * ``stream_ids`` maps every token position in ``x.shape[:-1]`` to a stream;
    * ``stream_axis`` identifies an explicit stream axis of size ``num_streams``;
    * ``stream_token_counts`` describes contiguous, possibly unequal blocks on
      the penultimate token axis.

    With no layout argument, an explicit axis 1 is detected for tensors such as
    ``[B, S, H, W, D]``; otherwise equal contiguous stream blocks are inferred
    on the penultimate axis for flattened ``[B, N, D]`` tensors.
    """

    if not isinstance(x, torch.Tensor) or x.ndim < 2:
        raise ValueError("x must be a Tensor with a final hidden dimension.")
    gains = torch.as_tensor(stream_gains, device=x.device)
    if gains.ndim != 1 or gains.numel() == 0:
        raise ValueError("stream_gains must be a non-empty one-dimensional tensor.")
    if not torch.isfinite(gains).all() or (gains <= 0).any():
        raise ValueError("stream_gains must be finite and positive.")

    selected_layouts = sum(value is not None for value in (stream_axis, stream_token_counts, stream_ids))
    if selected_layouts > 1:
        raise ValueError("Specify at most one of stream_axis, stream_token_counts, or stream_ids.")

    if stream_ids is not None:
        ids = torch.as_tensor(stream_ids, device=x.device)
        if ids.shape != x.shape[:-1]:
            raise ValueError(f"stream_ids must have shape {tuple(x.shape[:-1])}, got {tuple(ids.shape)}.")
        if ids.dtype == torch.bool or ids.is_floating_point():
            raise ValueError("stream_ids must contain integer stream indices.")
        ids = ids.long()
        if ids.numel() == 0 or ids.min() < 0 or ids.max() >= gains.numel():
            raise ValueError(f"stream_ids must lie in [0, {gains.numel()}).")
        return gains.index_select(0, ids.reshape(-1)).reshape(*ids.shape, 1)

    if stream_axis is not None:
        axis = int(stream_axis)
        if axis < 0:
            axis += x.ndim
        if not 0 <= axis < x.ndim - 1:
            raise ValueError("stream_axis must identify a non-hidden axis of x.")
        if x.shape[axis] != gains.numel():
            raise ValueError(f"x axis {axis} has size {x.shape[axis]}, expected {gains.numel()} streams.")
        shape = [1] * x.ndim
        shape[axis] = gains.numel()
        return gains.reshape(shape)

    if stream_token_counts is None and x.ndim >= 4 and x.shape[1] == gains.numel():
        shape = [1] * x.ndim
        shape[1] = gains.numel()
        return gains.reshape(shape)

    token_axis = x.ndim - 2
    num_tokens = x.shape[token_axis]
    if stream_token_counts is None:
        if num_tokens % gains.numel():
            raise ValueError(
                f"Token count {num_tokens} is not divisible by {gains.numel()} streams; "
                "provide stream_token_counts or stream_ids."
            )
        counts = torch.full(
            (gains.numel(),),
            num_tokens // gains.numel(),
            device=x.device,
            dtype=torch.long,
        )
    else:
        if len(stream_token_counts) != gains.numel():
            raise ValueError(f"Expected {gains.numel()} stream token counts, got {len(stream_token_counts)}.")
        counts = torch.as_tensor(stream_token_counts, device=x.device, dtype=torch.long)
        if (counts <= 0).any() or int(counts.sum().item()) != num_tokens:
            raise ValueError(f"stream_token_counts must be positive and sum to {num_tokens}.")
    gain_rows = torch.repeat_interleave(gains, counts)
    shape = [1] * x.ndim
    shape[token_axis] = num_tokens
    return gain_rows.reshape(shape)


class WAMFakeQuantLinear(nn.Module):
    """Drop-in reference Linear using calibrated WAM ``D``, gains, and clip."""

    def __init__(
        self,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale: torch.Tensor,
        stream_gains: torch.Tensor,
        clipping_ranges: torch.Tensor,
        token_fractions: torch.Tensor,
        weight_bits: int,
        activation_bits: int,
        input_rotation_signs: torch.Tensor | None = None,
        keep_output_dtype: str = "input",
        stream_axis: int | None = None,
        stream_token_counts: Sequence[int] | None = None,
        backend=None,
    ) -> None:
        super().__init__()
        if qweight.ndim != 2:
            raise ValueError("qweight must have shape [out_features, in_features].")
        out_features, in_features = qweight.shape
        d = torch.as_tensor(input_scale).detach().float()
        gains = torch.as_tensor(stream_gains).detach().float()
        clips = torch.as_tensor(clipping_ranges).detach().float()
        fractions = torch.as_tensor(token_fractions).detach().float()
        if d.shape != (in_features,) or not torch.isfinite(d).all() or (d <= 0).any():
            raise ValueError(f"input_scale must be a positive [{in_features}] vector.")
        if gains.ndim != 2 or not torch.isfinite(gains).all() or (gains <= 0).any():
            raise ValueError("stream_gains must have finite positive shape [timesteps, streams].")
        if clips.shape != (gains.shape[0],) or not torch.isfinite(clips).all() or (clips <= 0).any():
            raise ValueError("clipping_ranges must be positive with shape [timesteps].")
        if fractions.shape != gains.shape:
            raise ValueError("token_fractions must have the same shape as stream_gains.")
        if not torch.isfinite(fractions).all() or (fractions < 0).any():
            raise ValueError("token_fractions must be finite and non-negative.")
        normalized_fractions = fractions / fractions.sum(dim=1, keepdim=True).clamp_min(1e-8)
        conservation = (normalized_fractions * gains.log()).sum(dim=1)
        if not torch.allclose(conservation, torch.zeros_like(conservation), atol=1e-5, rtol=0):
            raise ValueError("stream_gains violate token-count-weighted log conservation.")
        if scales.ndim != 2 or scales.shape != (out_features, 1):
            raise ValueError(f"WAM expects symmetric per-output-channel scales [{out_features}, 1].")
        if not 2 <= int(weight_bits) <= 8 or not 2 <= int(activation_bits) <= 8:
            raise ValueError("weight_bits and activation_bits must be in [2, 8].")
        if stream_axis is not None and stream_token_counts is not None:
            raise ValueError("stream_axis and stream_token_counts are mutually exclusive.")
        counts = None if stream_token_counts is None else tuple(int(v) for v in stream_token_counts)
        if counts is not None and (len(counts) != gains.shape[1] or any(v <= 0 for v in counts)):
            raise ValueError("stream_token_counts must contain one positive count per stream.")
        rotation_signs = (
            None if input_rotation_signs is None else torch.as_tensor(input_rotation_signs).float()
        )
        if rotation_signs is not None:
            if rotation_signs.shape != (in_features,) or not torch.all(rotation_signs.abs() == 1):
                raise ValueError(f"input_rotation_signs must contain +/-1 with shape [{in_features}].")

        resolved_backend = get_backend("reference") if backend is None else backend
        stored_qweight = qweight.detach().contiguous().to(torch.int8)
        if getattr(resolved_backend, "packs_symmetric_int4", False):
            if weight_bits != 4:
                raise ValueError("The CUTLASS WAM backend requires INT4 weights.")
            stored_qweight = resolved_backend.pack_weight(stored_qweight)
        self.register_buffer("qweight", stored_qweight)
        self.register_buffer("scales", scales.detach().contiguous().float())
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.register_buffer("input_scale", d.contiguous())
        prepared_input_scale = None
        if hasattr(resolved_backend, "prepare_input_scale"):
            prepared_input_scale = resolved_backend.prepare_input_scale(d)
        self.register_buffer("_runtime_input_scale", prepared_input_scale, persistent=False)
        self.register_buffer("stream_gains", gains.contiguous())
        self.register_buffer("clipping_ranges", clips.contiguous())
        qmax = 2 ** (int(activation_bits) - 1) - 1
        self.register_buffer("activation_scales", (clips / qmax).contiguous())
        self.register_buffer("token_fractions", normalized_fractions.contiguous())
        self.register_buffer(
            "input_rotation_signs",
            None if rotation_signs is None else rotation_signs.detach().contiguous(),
        )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.keep_output_dtype = str(keep_output_dtype)
        self.stream_axis = stream_axis
        self.stream_token_counts = counts
        self.backend = resolved_backend
        self._timestep: int | None = None
        self._shared_timestep_state: _WAMTimestepState | None = None
        self._cached_stream_shape: tuple[int, ...] | None = None
        self._cached_stream_span: int | None = None
        wants_fused_preprocess = self.activation_bits == 4 and rotation_signs is not None
        if wants_fused_preprocess:
            self._fast_wam_runner = getattr(
                resolved_backend, "wam_stream_w4a4_preprocess_linear_prevalidated", None
            )
            self._fast_adaln_runner = getattr(
                resolved_backend, "wam_stream_w4a4_preprocess_adaln_linear_prevalidated", None
            )
            self._fast_gate_residual_runner = getattr(
                resolved_backend, "wam_stream_w4a4_preprocess_gate_residual_prevalidated", None
            )
        elif self.activation_bits == 4:
            self._fast_wam_runner = getattr(resolved_backend, "wam_stream_w4a4_linear_prevalidated", None)
            self._fast_adaln_runner = getattr(
                resolved_backend, "wam_stream_w4a4_adaln_linear_prevalidated", None
            )
            self._fast_gate_residual_runner = getattr(
                resolved_backend, "wam_stream_w4a4_gate_residual_prevalidated", None
            )
        else:
            self._fast_wam_runner = getattr(resolved_backend, "wam_stream_linear_prevalidated", None)
            self._fast_adaln_runner = getattr(resolved_backend, "wam_stream_adaln_linear_prevalidated", None)
            self._fast_gate_residual_runner = getattr(
                resolved_backend, "wam_stream_gate_residual_prevalidated", None
            )
        self._fuses_input_rotation = wants_fused_preprocess and self._fast_wam_runner is not None
        rotation_is_fused = rotation_signs is None or self._fuses_input_rotation
        self.supports_fused_adaln = self._fast_adaln_runner is not None and rotation_is_fused
        self.supports_fused_gate_residual = self._fast_gate_residual_runner is not None and rotation_is_fused

    @classmethod
    def from_calibration(
        cls,
        linear: nn.Linear,
        d_result: WAMDCalibrationResult,
        gamma_result: WAMGammaCalibrationResult,
        *,
        weight_bits: int,
        activation_bits: int,
        keep_output_dtype: str = "input",
        stream_axis: int | None = None,
        stream_token_counts: Sequence[int] | None = None,
        backend=None,
    ) -> WAMFakeQuantLinear:
        """Construct a runtime module from the outputs of Parts 3 and 4."""

        if d_result.layer != gamma_result.layer:
            raise ValueError("D and gamma calibration results refer to different layers.")
        if d_result.module_name != gamma_result.module_name:
            raise ValueError("D and gamma calibration results refer to different modules.")
        if (
            gamma_result.stream_indices
            and len(gamma_result.stream_indices) != gamma_result.stream_gains.shape[1]
        ):
            raise ValueError("Gamma result stream indices do not match its gain table.")
        quantized = d_result.quantized_weight
        if quantized.zeros is not None:
            raise ValueError("WAM runtime currently requires symmetric weights.")
        if linear.weight.shape != quantized.qweight.shape:
            raise ValueError("Source Linear and quantized weight have different shapes.")
        qmax = 2 ** (int(activation_bits) - 1) - 1
        expected_scales = gamma_result.clipping_ranges / qmax
        if not torch.allclose(
            gamma_result.activation_scales,
            expected_scales,
            atol=1e-7,
            rtol=1e-6,
        ):
            raise ValueError("Gamma result activation scales do not match clipping ranges.")
        return cls(
            qweight=quantized.qweight,
            scales=quantized.scales,
            bias=linear.bias,
            input_scale=d_result.input_scale,
            stream_gains=gamma_result.stream_gains,
            clipping_ranges=gamma_result.clipping_ranges,
            token_fractions=gamma_result.token_fractions,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            keep_output_dtype=keep_output_dtype,
            stream_axis=stream_axis,
            stream_token_counts=stream_token_counts,
            backend=backend,
        )

    @property
    def num_denoise_calls(self) -> int:
        return int(self.stream_gains.shape[0])

    @property
    def num_streams(self) -> int:
        return int(self.stream_gains.shape[1])

    @property
    def timestep(self) -> int | None:
        if self._shared_timestep_state is not None:
            return self._shared_timestep_state.value
        return self._timestep

    def set_timestep(self, timestep: int) -> None:
        if self._shared_timestep_state is not None:
            self._shared_timestep_state.set(timestep)
            return
        timestep = int(timestep)
        if not 0 <= timestep < self.num_denoise_calls:
            raise IndexError(f"timestep {timestep} is outside [0, {self.num_denoise_calls}).")
        self._timestep = timestep

    def clear_timestep(self) -> None:
        if self._shared_timestep_state is not None:
            self._shared_timestep_state.value = None
            return
        self._timestep = None

    def forward(self, x: torch.Tensor, stream_ids: torch.Tensor | None = None) -> torch.Tensor:
        shared_state = self._shared_timestep_state
        timestep = shared_state.value if shared_state is not None else self._timestep
        if timestep is None:
            raise RuntimeError("Set the WAM denoise-call index before running inference.")
        if (
            stream_ids is None
            and (self.input_rotation_signs is None or self._fuses_input_rotation)
            and self._cached_stream_span is not None
            and self._fast_wam_runner is not None
        ):
            args = (
                x,
                self.qweight,
                self.scales,
                self.bias,
                self._runtime_input_scale,
                self.activation_scales,
                self.stream_gains,
                self._cached_stream_span,
                timestep,
            )
            if self._fuses_input_rotation:
                return self._fast_wam_runner(*args, self.input_rotation_signs)
            return self._fast_wam_runner(*args)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input hidden size {self.in_features}, got {x.shape[-1]}.")
        if self.input_rotation_signs is not None and not self._fuses_input_rotation:
            x = randomized_hadamard_transform(x, self.input_rotation_signs)
        input_shape = tuple(x.shape[:-1])
        if stream_ids is None and input_shape == self._cached_stream_shape:
            stream_span = self._cached_stream_span
        else:
            stream_span = self._cutlass_stream_span(x, stream_ids)
            if stream_ids is None:
                self._cached_stream_shape = input_shape
                self._cached_stream_span = stream_span
        if stream_span is not None and self._fast_wam_runner is not None:
            args = (
                x,
                self.qweight,
                self.scales,
                self.bias,
                self._runtime_input_scale,
                self.activation_scales,
                self.stream_gains,
                stream_span,
                timestep,
            )
            if self._fuses_input_rotation:
                return self._fast_wam_runner(*args, self.input_rotation_signs)
            return self._fast_wam_runner(*args)
        if self.input_rotation_signs is not None and self._fuses_input_rotation:
            # Irregular stream IDs cannot use the compact fused stream mapper.
            # Preserve correctness by falling back to the standalone transform.
            x = randomized_hadamard_transform(x, self.input_rotation_signs)
        gains = self.stream_gains[timestep]
        activation_scale = self.activation_scales[timestep]
        if stream_span is not None and hasattr(self.backend, "wam_stream_linear"):
            return self.backend.wam_stream_linear(
                x=x,
                qweight=self.qweight,
                scales=self.scales,
                bias=self.bias,
                input_scale=self.input_scale,
                activation_scale=activation_scale,
                stream_gains=gains,
                stream_span=stream_span,
                activation_bits=self.activation_bits,
                keep_output_dtype=self.keep_output_dtype,
            )
        gain_rows = expand_wam_stream_gains(
            x,
            gains,
            stream_axis=self.stream_axis,
            stream_token_counts=self.stream_token_counts,
            stream_ids=stream_ids,
        ).to(dtype=x.dtype)
        if hasattr(self.backend, "wam_linear"):
            return self.backend.wam_linear(
                x=x,
                qweight=self.qweight,
                scales=self.scales,
                bias=self.bias,
                input_scale=self.input_scale,
                activation_scale=activation_scale,
                row_gains=gain_rows,
                activation_bits=self.activation_bits,
                keep_output_dtype=self.keep_output_dtype,
            )
        input_scale = self.input_scale.to(device=x.device, dtype=x.dtype)
        x_prime = x / input_scale
        scaled = x_prime * gain_rows
        quantized = fake_quant_activation(
            scaled,
            bits=self.activation_bits,
            symmetric=True,
            granularity="per_tensor",
            static_scale=activation_scale,
        )
        reconstructed = quantized / gain_rows
        return self.backend.linear(
            x=reconstructed,
            qweight=self.qweight,
            scales=self.scales,
            zeros=None,
            bias=self.bias,
            group_size=-1,
            symmetric=True,
            input_scale=None,
            activation_bits=self.activation_bits,
            activation_granularity="per_tensor",
            activation_scale=None,
            activation_zero=None,
            fake_quant_activation_enabled=False,
            keep_output_dtype=self.keep_output_dtype,
        )

    @staticmethod
    def _group_span(x: torch.Tensor, grouped: torch.Tensor, width: int, name: str) -> int:
        rows = x.numel() // x.shape[-1]
        grouped_rows = grouped.numel() // width
        if grouped.shape[-1] != width or grouped_rows <= 0 or rows % grouped_rows:
            raise ValueError(f"{name} must provide one width-{width} row per contiguous output group.")
        return rows // grouped_rows

    def forward_adaln(
        self,
        x: torch.Tensor,
        adaln_scale: torch.Tensor,
        adaln_shift: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """CUTLASS path for LayerNorm + AdaLN + scheduled WAM A4/A8 + Linear."""

        if not self.supports_fused_adaln:
            raise RuntimeError("The selected WAM backend does not support fused AdaLN.")
        timestep = self.timestep
        if timestep is None:
            raise RuntimeError("Set the WAM denoise-call index before running inference.")
        modulation_span = self._group_span(x, adaln_scale, self.in_features, "adaln_scale")
        if adaln_shift.shape != adaln_scale.shape:
            raise ValueError("adaln_scale and adaln_shift must have matching shapes.")
        stream_span = self._cutlass_stream_span(x, None)
        if stream_span is None:
            raise ValueError("Fused WAM AdaLN requires a contiguous semantic-stream layout.")
        args = (
            x,
            adaln_scale,
            adaln_shift,
            modulation_span,
            float(epsilon),
            self.qweight,
            self.scales,
            self.bias,
            self._runtime_input_scale,
            self.activation_scales,
            self.stream_gains,
            stream_span,
            timestep,
        )
        if self._fuses_input_rotation:
            return self._fast_adaln_runner(*args, self.input_rotation_signs)
        return self._fast_adaln_runner(*args)

    def forward_gate_residual(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """CUTLASS path for scheduled WAM Linear + AdaLN gate + residual."""

        if not self.supports_fused_gate_residual:
            raise RuntimeError("The selected WAM backend does not support fused gated residual.")
        timestep = self.timestep
        if timestep is None:
            raise RuntimeError("Set the WAM denoise-call index before running inference.")
        stream_span = self._cutlass_stream_span(x, None)
        if stream_span is None:
            raise ValueError("Fused WAM residual requires a contiguous semantic-stream layout.")
        gate_span = self._group_span(x, gate, self.out_features, "gate")
        args = (
            x,
            self.qweight,
            self.scales,
            self.bias,
            self._runtime_input_scale,
            self.activation_scales,
            self.stream_gains,
            stream_span,
            timestep,
            residual,
            gate,
            gate_span,
        )
        if self._fuses_input_rotation:
            return self._fast_gate_residual_runner(*args, self.input_rotation_signs)
        return self._fast_gate_residual_runner(*args)

    def _cutlass_stream_span(self, x: torch.Tensor, stream_ids: torch.Tensor | None) -> int | None:
        """Return the contiguous rows per stream for the fused WAM CUDA path."""

        if stream_ids is not None:
            return None
        if self.num_streams == 1:
            return 1
        if self.stream_axis is not None:
            axis = int(self.stream_axis)
            if axis < 0:
                axis += x.ndim
            if not 0 <= axis < x.ndim - 1 or x.shape[axis] != self.num_streams:
                return None
            return math.prod(x.shape[axis + 1 : -1])
        if self.stream_token_counts is not None:
            if len(set(self.stream_token_counts)) != 1:
                return None
            if sum(self.stream_token_counts) != x.shape[-2]:
                return None
            return self.stream_token_counts[0]
        if x.ndim >= 4 and x.shape[1] == self.num_streams:
            return math.prod(x.shape[2:-1])
        tokens = x.shape[-2]
        if tokens % self.num_streams:
            return None
        return tokens // self.num_streams

    def extra_repr(self) -> str:
        backend_name = getattr(self.backend, "name", "reference")
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"W{self.weight_bits}A{self.activation_bits}, "
            f"timesteps={self.num_denoise_calls}, streams={self.num_streams}, "
            f"preprocess={self.input_rotation_signs is not None}, WAM {backend_name}"
        )


def iter_wam_fake_quant_linears(model: nn.Module) -> Iterator[tuple[str, WAMFakeQuantLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, WAMFakeQuantLinear):
            yield name, module


def set_wam_timestep(model: nn.Module, timestep: int) -> int:
    """Set one denoise-call lookup index on every WAM Linear in ``model``."""

    shared_state = getattr(model, "_wam_timestep_state", None)
    if isinstance(shared_state, _WAMTimestepState):
        shared_state.set(timestep)
        return shared_state.module_count
    count = 0
    for _name, module in iter_wam_fake_quant_linears(model):
        module.set_timestep(timestep)
        count += 1
    if count == 0:
        raise RuntimeError("No WAMFakeQuantLinear modules were found in the model.")
    return count


def clear_wam_timestep(model: nn.Module) -> int:
    shared_state = getattr(model, "_wam_timestep_state", None)
    if isinstance(shared_state, _WAMTimestepState):
        shared_state.value = None
        return shared_state.module_count
    count = 0
    for _name, module in iter_wam_fake_quant_linears(model):
        module.clear_timestep()
        count += 1
    return count


def bind_wam_timestep_state(model: nn.Module) -> int:
    """Bind all WAM Linears to one O(1) network-level timestep state."""

    modules = tuple(module for _name, module in iter_wam_fake_quant_linears(model))
    if not modules:
        raise RuntimeError("No WAMFakeQuantLinear modules were found in the model.")
    num_denoise_calls = modules[0].num_denoise_calls
    if any(module.num_denoise_calls != num_denoise_calls for module in modules):
        raise ValueError("All WAM modules must use the same denoise-call schedule.")
    state = _WAMTimestepState(num_denoise_calls, len(modules))
    for module in modules:
        module._shared_timestep_state = state
    setattr(model, "_wam_timestep_state", state)
    return len(modules)


__all__ = [
    "WAMFakeQuantLinear",
    "bind_wam_timestep_state",
    "clear_wam_timestep",
    "expand_wam_stream_gains",
    "iter_wam_fake_quant_linears",
    "set_wam_timestep",
]
