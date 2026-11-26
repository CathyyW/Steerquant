"""End-to-end calibration, checkpointing, and model integration for WAM."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .algorithms.wam import build_wam_sensitivity_weights
from .algorithms.wam_d import WAMDCalibrationConfig, calibrate_wam_d_for_layer
from .algorithms.wam_gamma import WAMGammaCalibrationConfig, calibrate_wam_gamma_for_layer
from .wam_cache import WAMActivationCache, WAMCacheSite
from .wam_runtime import WAMFakeQuantLinear, bind_wam_timestep_state, set_wam_timestep

WAM_CHECKPOINT_FORMAT = "cosmos_policy_wam_v1"


@dataclass(frozen=True)
class WAMSensitivityField:
    """Observation-aggregated WAM field and its module/stream topology."""

    values: torch.Tensor
    active_stream_mask: torch.Tensor
    module_names: tuple[str, ...]
    slot_aligned: tuple[bool, ...]
    observation_counts: torch.Tensor

    def validate(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("Sensitivity values must have shape [layers, timesteps, streams].")
        if self.active_stream_mask.shape != self.values.shape:
            raise ValueError("active_stream_mask must have the same shape as sensitivity values.")
        if self.observation_counts.shape != self.values.shape:
            raise ValueError("observation_counts must have the same shape as sensitivity values.")
        if len(self.module_names) != self.values.shape[0] or len(self.slot_aligned) != self.values.shape[0]:
            raise ValueError("Module metadata must contain one item per sensitivity layer.")
        if len(set(self.module_names)) != len(self.module_names):
            raise ValueError("Sensitivity module names must be unique.")
        if not torch.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("Sensitivity values must be finite and non-negative.")
        if not self.active_stream_mask.any(dim=2).all():
            raise ValueError("Every layer and timestep must have an active stream.")
        active_counts = self.observation_counts[self.active_stream_mask]
        if active_counts.numel() == 0 or (active_counts <= 0).any():
            raise ValueError("Every active sensitivity cell must contain observations.")


def load_wam_sensitivity_field_csv(
    path: str | Path,
    *,
    num_denoise_calls: int = 5,
    num_semantic_streams: int = 9,
    quantization_variant: str = "joint_wa",
    score_column: str = "predicted_standardized_action_rmse",
    expected_observations: int | None = None,
) -> WAMSensitivityField:
    """Stream observation-level scores into the dataset RMS sensitivity field.

    Each active cell is aggregated as ``sqrt(mean_x(score(x)**2))``.  Global
    text-context K/V rows (``slot_aligned=False``) occupy stream column zero and
    mark all other semantic-stream columns inactive; they are never mislabeled
    as one of the nine WAM slots.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if int(num_denoise_calls) <= 0 or int(num_semantic_streams) <= 0:
        raise ValueError("Denoise-call and semantic-stream counts must be positive.")

    sums: dict[tuple[int, int, int, int], float] = {}
    counts: dict[tuple[int, int, int, int], int] = {}
    metadata: dict[tuple[int, int], tuple[str, bool]] = {}
    required = {
        "block_index",
        "linear_index",
        "module_path",
        "denoise_call",
        "slot_aligned",
        "slot_index",
        "quantization_variant",
        score_column,
    }
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Sensitivity CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["quantization_variant"] != quantization_variant:
                continue
            raw_score = row[score_column].strip()
            if not raw_score:
                continue
            score = float(raw_score)
            if not math.isfinite(score) or score < 0:
                raise ValueError("Observation sensitivity scores must be finite and non-negative.")
            block = int(row["block_index"])
            linear = int(row["linear_index"])
            timestep = int(row["denoise_call"])
            if not 0 <= timestep < int(num_denoise_calls):
                raise ValueError(f"Denoise call {timestep} is outside the configured schedule.")
            aligned_text = row["slot_aligned"].strip().lower()
            if aligned_text not in {"true", "1", "yes", "false", "0", "no"}:
                raise ValueError(f"Invalid slot_aligned value {row['slot_aligned']!r}.")
            aligned = aligned_text in {"true", "1", "yes"}
            slot = int(row["slot_index"]) if aligned else 0
            if not 0 <= slot < int(num_semantic_streams):
                raise ValueError(f"Semantic stream {slot} is outside the configured field.")
            layer_key = (block, linear)
            module_name = f"blocks.{block - 1}.{row['module_path']}"
            previous = metadata.setdefault(layer_key, (module_name, aligned))
            if previous != (module_name, aligned):
                raise ValueError(f"Inconsistent metadata for block/linear {layer_key}.")
            key = (block, linear, timestep, slot)
            sums[key] = sums.get(key, 0.0) + score * score
            counts[key] = counts.get(key, 0) + 1

    if not metadata:
        raise ValueError("No matching observation-level sensitivity rows were found.")
    layer_keys = sorted(metadata)
    layer_lookup = {key: index for index, key in enumerate(layer_keys)}
    shape = (len(layer_keys), int(num_denoise_calls), int(num_semantic_streams))
    values = torch.zeros(shape, dtype=torch.float32)
    observation_counts = torch.zeros(shape, dtype=torch.long)
    active_mask = torch.zeros(shape, dtype=torch.bool)
    module_names: list[str] = []
    slot_aligned: list[bool] = []
    for layer_key in layer_keys:
        module_name, aligned = metadata[layer_key]
        module_names.append(module_name)
        slot_aligned.append(aligned)
        layer = layer_lookup[layer_key]
        active_mask[layer, :, :] = aligned
        if not aligned:
            active_mask[layer, :, 0] = True
    for (block, linear, timestep, stream), total in sums.items():
        layer = layer_lookup[(block, linear)]
        count = counts[(block, linear, timestep, stream)]
        values[layer, timestep, stream] = math.sqrt(total / count)
        observation_counts[layer, timestep, stream] = count

    if expected_observations is not None:
        active_counts = observation_counts[active_mask]
        if not torch.equal(
            active_counts,
            torch.full_like(active_counts, int(expected_observations)),
        ):
            unique = sorted(set(int(value) for value in active_counts.tolist()))
            raise ValueError(
                f"Expected {expected_observations} observations in every active cell, got {unique}."
            )
    field = WAMSensitivityField(
        values=values,
        active_stream_mask=active_mask,
        module_names=tuple(module_names),
        slot_aligned=tuple(slot_aligned),
        observation_counts=observation_counts,
    )
    field.validate()
    return field


@dataclass(frozen=True)
class WAMLinearCheckpoint:
    layer: int
    module_name: str
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    input_scale: torch.Tensor
    stream_gains: torch.Tensor
    clipping_ranges: torch.Tensor
    token_fractions: torch.Tensor
    stream_indices: tuple[int, ...]
    d_initial_hard_loss: float
    d_final_hard_loss: float
    gamma_initial_hard_losses: tuple[float, ...]
    gamma_final_hard_losses: tuple[float, ...]
    input_rotation_signs: torch.Tensor | None = None

    def validate(self, *, num_denoise_calls: int) -> None:
        if self.qweight.ndim != 2:
            raise ValueError("WAM checkpoint qweight must be two-dimensional.")
        out_features, in_features = self.qweight.shape
        num_streams = len(self.stream_indices)
        if num_streams == 0 or len(set(self.stream_indices)) != num_streams:
            raise ValueError("WAM checkpoint stream indices must be non-empty and unique.")
        if self.stream_indices != tuple(range(num_streams)):
            raise ValueError(
                "The current implicit runtime layout requires contiguous stream indices from zero."
            )
        if self.weight_scales.shape != (out_features, 1):
            raise ValueError("WAM checkpoint weight scales must be per output channel.")
        if self.input_scale.shape != (in_features,) or (self.input_scale <= 0).any():
            raise ValueError("WAM checkpoint input scale has an invalid shape or value.")
        if self.input_rotation_signs is not None:
            if self.input_rotation_signs.shape != (in_features,):
                raise ValueError("WAM checkpoint Preprocess signs have an invalid shape.")
            if not torch.isfinite(self.input_rotation_signs).all() or not torch.all(
                self.input_rotation_signs.abs() == 1
            ):
                raise ValueError("WAM checkpoint Preprocess signs must contain only finite +/-1 values.")
        if self.stream_gains.shape != (num_denoise_calls, num_streams):
            raise ValueError("WAM checkpoint gain lookup has an invalid shape.")
        if self.clipping_ranges.shape != (num_denoise_calls,) or (self.clipping_ranges <= 0).any():
            raise ValueError("WAM checkpoint clipping lookup has an invalid shape or value.")
        if self.token_fractions.shape != self.stream_gains.shape:
            raise ValueError("WAM checkpoint token fractions must match its gain lookup.")
        if (self.token_fractions < 0).any() or not torch.allclose(
            self.token_fractions.sum(dim=1),
            torch.ones(num_denoise_calls),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("WAM checkpoint token fractions must be non-negative and sum to one.")
        if not all(
            torch.isfinite(tensor).all()
            for tensor in (
                self.weight_scales,
                self.input_scale,
                self.stream_gains,
                self.clipping_ranges,
                self.token_fractions,
            )
        ):
            raise ValueError("WAM checkpoint contains non-finite tensors.")
        conservation = (self.token_fractions * self.stream_gains.log()).sum(dim=1)
        if not torch.allclose(conservation, torch.zeros_like(conservation), atol=1e-5, rtol=0):
            raise ValueError("WAM checkpoint gains violate weighted-log conservation.")


@dataclass(frozen=True)
class WAMCalibrationCheckpoint:
    weight_bits: int
    activation_bits: int
    num_denoise_calls: int
    sensitivity: torch.Tensor
    active_stream_mask: torch.Tensor
    timestep_weights: torch.Tensor
    rho: float
    eps: float
    u_min: float | None
    u_max: float | None
    d_config: dict[str, Any]
    gamma_config: dict[str, Any]
    linears: tuple[WAMLinearCheckpoint, ...]

    def validate(self) -> None:
        if not 2 <= int(self.weight_bits) <= 8 or not 2 <= int(self.activation_bits) <= 8:
            raise ValueError("WAM checkpoint bit widths must be in [2, 8].")
        if self.sensitivity.ndim != 3 or self.active_stream_mask.shape != self.sensitivity.shape:
            raise ValueError("WAM checkpoint sensitivity field has an invalid shape.")
        if self.sensitivity.shape[1] != self.num_denoise_calls:
            raise ValueError("WAM checkpoint sensitivity and denoise schedule disagree.")
        if self.timestep_weights.shape != (self.num_denoise_calls,):
            raise ValueError("WAM checkpoint timestep weights have an invalid shape.")
        if len(self.linears) != self.sensitivity.shape[0]:
            raise ValueError("WAM checkpoint must contain one Linear entry per sensitivity layer.")
        names = [entry.module_name for entry in self.linears]
        if len(set(names)) != len(names):
            raise ValueError("WAM checkpoint module names must be unique.")
        if [entry.layer for entry in self.linears] != list(range(len(self.linears))):
            raise ValueError("WAM checkpoint Linear entries must follow contiguous layer order.")
        for entry in self.linears:
            entry.validate(num_denoise_calls=self.num_denoise_calls)
            expected_streams: tuple[int, ...] | None = None
            for timestep in range(self.num_denoise_calls):
                active = tuple(
                    int(value)
                    for value in torch.nonzero(self.active_stream_mask[entry.layer, timestep])
                    .flatten()
                    .tolist()
                )
                if expected_streams is None:
                    expected_streams = active
                elif active != expected_streams:
                    raise ValueError("Active stream layout must remain fixed across timesteps.")
            if entry.stream_indices != expected_streams:
                raise ValueError(f"Checkpoint stream layout does not match sensitivity layer {entry.layer}.")
            qmax = 2 ** (int(self.weight_bits) - 1) - 1
            if entry.qweight.dtype != torch.int8 or (entry.qweight.to(torch.int16).abs() > qmax).any():
                raise ValueError("Checkpoint integer weights exceed the configured symmetric range.")

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": WAM_CHECKPOINT_FORMAT,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "num_denoise_calls": self.num_denoise_calls,
            "sensitivity": self.sensitivity.cpu(),
            "active_stream_mask": self.active_stream_mask.cpu(),
            "timestep_weights": self.timestep_weights.cpu(),
            "rho": self.rho,
            "eps": self.eps,
            "u_min": self.u_min,
            "u_max": self.u_max,
            "d_config": self.d_config,
            "gamma_config": self.gamma_config,
            "linears": [
                {
                    "layer": entry.layer,
                    "module_name": entry.module_name,
                    "qweight": entry.qweight.cpu(),
                    "weight_scales": entry.weight_scales.cpu(),
                    "input_scale": entry.input_scale.cpu(),
                    "stream_gains": entry.stream_gains.cpu(),
                    "clipping_ranges": entry.clipping_ranges.cpu(),
                    "token_fractions": entry.token_fractions.cpu(),
                    "stream_indices": entry.stream_indices,
                    "d_initial_hard_loss": entry.d_initial_hard_loss,
                    "d_final_hard_loss": entry.d_final_hard_loss,
                    "gamma_initial_hard_losses": entry.gamma_initial_hard_losses,
                    "gamma_final_hard_losses": entry.gamma_final_hard_losses,
                    "input_rotation_signs": (
                        None if entry.input_rotation_signs is None else entry.input_rotation_signs.cpu()
                    ),
                }
                for entry in self.linears
            ],
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> WAMCalibrationCheckpoint:
        if not isinstance(payload, dict) or payload.get("format") != WAM_CHECKPOINT_FORMAT:
            raise ValueError(f"Expected WAM checkpoint format {WAM_CHECKPOINT_FORMAT!r}.")
        entries = tuple(
            WAMLinearCheckpoint(
                layer=int(stored["layer"]),
                module_name=str(stored["module_name"]),
                qweight=torch.as_tensor(stored["qweight"], dtype=torch.int8).cpu(),
                weight_scales=torch.as_tensor(stored["weight_scales"], dtype=torch.float32).cpu(),
                input_scale=torch.as_tensor(stored["input_scale"], dtype=torch.float32).cpu(),
                stream_gains=torch.as_tensor(stored["stream_gains"], dtype=torch.float32).cpu(),
                clipping_ranges=torch.as_tensor(stored["clipping_ranges"], dtype=torch.float32).cpu(),
                token_fractions=torch.as_tensor(stored["token_fractions"], dtype=torch.float32).cpu(),
                stream_indices=tuple(int(value) for value in stored["stream_indices"]),
                d_initial_hard_loss=float(stored["d_initial_hard_loss"]),
                d_final_hard_loss=float(stored["d_final_hard_loss"]),
                gamma_initial_hard_losses=tuple(
                    float(value) for value in stored["gamma_initial_hard_losses"]
                ),
                gamma_final_hard_losses=tuple(float(value) for value in stored["gamma_final_hard_losses"]),
                input_rotation_signs=(
                    None
                    if stored.get("input_rotation_signs") is None
                    else torch.as_tensor(stored["input_rotation_signs"], dtype=torch.float32).cpu()
                ),
            )
            for stored in payload["linears"]
        )
        checkpoint = cls(
            weight_bits=int(payload["weight_bits"]),
            activation_bits=int(payload["activation_bits"]),
            num_denoise_calls=int(payload["num_denoise_calls"]),
            sensitivity=torch.as_tensor(payload["sensitivity"], dtype=torch.float32).cpu(),
            active_stream_mask=torch.as_tensor(payload["active_stream_mask"], dtype=torch.bool).cpu(),
            timestep_weights=torch.as_tensor(payload["timestep_weights"], dtype=torch.float32).cpu(),
            rho=float(payload["rho"]),
            eps=float(payload["eps"]),
            u_min=None if payload["u_min"] is None else float(payload["u_min"]),
            u_max=None if payload["u_max"] is None else float(payload["u_max"]),
            d_config=dict(payload["d_config"]),
            gamma_config=dict(payload["gamma_config"]),
            linears=entries,
        )
        checkpoint.validate()
        return checkpoint

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), resolved)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> WAMCalibrationCheckpoint:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(resolved, map_location="cpu")
        return cls.from_state_dict(payload)


def _canonical_module_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _resolve_named_linear(model: nn.Module, module_name: str) -> tuple[str, nn.Linear]:
    canonical = _canonical_module_name(module_name)
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if _canonical_module_name(name) == canonical and isinstance(module, nn.Linear)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Linear matching canonical module {module_name!r}, found {len(matches)}."
        )
    return matches[0]


def _resolve_linear(model: nn.Module, module_name: str) -> nn.Linear:
    _actual_name, module = _resolve_named_linear(model, module_name)
    return module


def build_wam_cache_sites(
    model: nn.Module,
    sensitivity_field: WAMSensitivityField,
    *,
    layer_indices: Sequence[int] | None = None,
) -> list[WAMCacheSite]:
    """Resolve sensitivity modules into hook sites, optionally block by block."""

    sensitivity_field.validate()
    selected = range(len(sensitivity_field.module_names)) if layer_indices is None else layer_indices
    sites: list[WAMCacheSite] = []
    for layer_value in selected:
        layer = int(layer_value)
        if not 0 <= layer < len(sensitivity_field.module_names):
            raise IndexError(f"WAM layer index {layer} is outside the sensitivity field.")
        active_by_timestep = [
            tuple(
                int(value)
                for value in torch.nonzero(sensitivity_field.active_stream_mask[layer, timestep])
                .flatten()
                .tolist()
            )
            for timestep in range(sensitivity_field.values.shape[1])
        ]
        if any(indices != active_by_timestep[0] for indices in active_by_timestep[1:]):
            raise ValueError("Cache collection requires a fixed stream layout across timesteps.")
        module_name = sensitivity_field.module_names[layer]
        _actual_name, module = _resolve_named_linear(model, module_name)
        sites.append(
            WAMCacheSite(
                layer=layer,
                module_name=module_name,
                module=module,
                slot_aligned=sensitivity_field.slot_aligned[layer],
                stream_indices=active_by_timestep[0],
            )
        )
    return sites


def calibrate_wam_model(
    model: nn.Module,
    *,
    cache: WAMActivationCache,
    sensitivity_field: WAMSensitivityField,
    timestep_weights: torch.Tensor | None = None,
    rho: float = 1.0,
    eps: float = 1e-8,
    u_min: float | None = None,
    u_max: float | None = None,
    d_config: WAMDCalibrationConfig | None = None,
    gamma_config: WAMGammaCalibrationConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    release_cache_after_layer: bool = False,
    input_rotation_signs: Mapping[str, torch.Tensor] | None = None,
) -> WAMCalibrationCheckpoint:
    """Calibrate all cached Linears sequentially and return one WAM checkpoint."""

    sensitivity_field.validate()
    d_config = WAMDCalibrationConfig() if d_config is None else d_config
    gamma_config = WAMGammaCalibrationConfig() if gamma_config is None else gamma_config
    d_config.validate()
    gamma_config.validate()
    if d_config.activation_bits != gamma_config.activation_bits:
        raise ValueError("D and gamma calibration must use the same activation bit width.")
    if cache.config.num_layers != sensitivity_field.values.shape[0]:
        raise ValueError("Cache and sensitivity field have different layer counts.")
    if cache.config.num_denoise_calls != sensitivity_field.values.shape[1]:
        raise ValueError("Cache and sensitivity field have different denoise-call counts.")
    if cache.config.num_streams != sensitivity_field.values.shape[2]:
        raise ValueError("Cache and sensitivity field have different semantic-stream counts.")

    derived = build_wam_sensitivity_weights(
        sensitivity_field.values,
        timestep_weights=timestep_weights,
        rho=rho,
        eps=eps,
        u_min=u_min,
        u_max=u_max,
        active_stream_mask=sensitivity_field.active_stream_mask,
    )
    entries: list[WAMLinearCheckpoint] = []
    total = len(sensitivity_field.module_names)
    for layer, module_name in enumerate(sensitivity_field.module_names):
        linear = _resolve_linear(model, module_name)
        if progress is not None:
            progress(layer, total, module_name)
        d_result = calibrate_wam_d_for_layer(
            linear.weight,
            layer=layer,
            cache=cache,
            sensitivity_weights=derived,
            config=d_config,
        )
        gamma_result = calibrate_wam_gamma_for_layer(
            layer=layer,
            cache=cache,
            sensitivity_weights=derived,
            d_result=d_result,
            config=gamma_config,
        )
        entries.append(
            WAMLinearCheckpoint(
                layer=layer,
                module_name=module_name,
                qweight=d_result.quantized_weight.qweight.detach().cpu(),
                weight_scales=d_result.quantized_weight.scales.detach().cpu(),
                input_scale=d_result.input_scale.detach().cpu(),
                stream_gains=gamma_result.stream_gains.detach().cpu(),
                clipping_ranges=gamma_result.clipping_ranges.detach().cpu(),
                token_fractions=gamma_result.token_fractions.detach().cpu(),
                stream_indices=gamma_result.stream_indices,
                d_initial_hard_loss=d_result.initial_hard_loss,
                d_final_hard_loss=d_result.final_hard_loss,
                gamma_initial_hard_losses=gamma_result.initial_hard_losses,
                gamma_final_hard_losses=gamma_result.final_hard_losses,
                input_rotation_signs=(
                    None
                    if input_rotation_signs is None
                    else input_rotation_signs[module_name].detach().float().cpu()
                ),
            )
        )
        if release_cache_after_layer:
            cache.release_layer(layer)
    checkpoint = WAMCalibrationCheckpoint(
        weight_bits=d_config.weight_bits,
        activation_bits=d_config.activation_bits,
        num_denoise_calls=cache.config.num_denoise_calls,
        sensitivity=sensitivity_field.values.detach().cpu(),
        active_stream_mask=sensitivity_field.active_stream_mask.detach().cpu(),
        timestep_weights=derived.timestep_weights.detach().cpu(),
        rho=float(rho),
        eps=float(eps),
        u_min=u_min,
        u_max=u_max,
        d_config=asdict(d_config),
        gamma_config=asdict(gamma_config),
        linears=tuple(entries),
    )
    checkpoint.validate()
    return checkpoint


def calibrate_wam_model_grouped(
    model: nn.Module,
    *,
    sensitivity_field: WAMSensitivityField,
    cache_provider: Callable[[tuple[int, ...]], WAMActivationCache],
    layers_per_group: int = 10,
    timestep_weights: torch.Tensor | None = None,
    rho: float = 1.0,
    eps: float = 1e-8,
    u_min: float | None = None,
    u_max: float | None = None,
    d_config: WAMDCalibrationConfig | None = None,
    gamma_config: WAMGammaCalibrationConfig | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    input_rotation_signs: Mapping[str, torch.Tensor] | None = None,
    completed_linears: Mapping[int, WAMLinearCheckpoint] | None = None,
    layer_completed: Callable[[WAMLinearCheckpoint], None] | None = None,
) -> WAMCalibrationCheckpoint:
    """Calibrate in cache groups so one block's activations can be freed at a time.

    The provider receives global layer indices and must return a cache whose
    configuration still describes the full sensitivity topology, but it only
    needs cells for the requested layers.  With Cosmos-Policy's ten Linears per
    block, the default ``layers_per_group=10`` implements block-wise caching.
    """

    sensitivity_field.validate()
    if int(layers_per_group) <= 0:
        raise ValueError("layers_per_group must be positive.")
    d_config = WAMDCalibrationConfig() if d_config is None else d_config
    gamma_config = WAMGammaCalibrationConfig() if gamma_config is None else gamma_config
    d_config.validate()
    gamma_config.validate()
    if d_config.activation_bits != gamma_config.activation_bits:
        raise ValueError("D and gamma calibration must use the same activation bit width.")
    derived = build_wam_sensitivity_weights(
        sensitivity_field.values,
        timestep_weights=timestep_weights,
        rho=rho,
        eps=eps,
        u_min=u_min,
        u_max=u_max,
        active_stream_mask=sensitivity_field.active_stream_mask,
    )
    total = len(sensitivity_field.module_names)
    restored = dict(completed_linears or {})
    for layer, entry in restored.items():
        if not 0 <= layer < total or entry.layer != layer or entry.module_name != sensitivity_field.module_names[layer]:
            raise ValueError("Completed Linear identity does not match sensitivity topology")
        entry.validate(num_denoise_calls=sensitivity_field.values.shape[1])
        if entry.qweight.shape != _resolve_linear(model, entry.module_name).weight.shape:
            raise ValueError("Completed Linear shape does not match model")
    entries: list[WAMLinearCheckpoint] = list(restored.values())
    for start in range(0, total, int(layers_per_group)):
        group = tuple(layer for layer in range(start, min(start + int(layers_per_group), total)) if layer not in restored)
        if not group:
            continue
        cache = cache_provider(group)
        if (
            cache.config.num_layers != sensitivity_field.values.shape[0]
            or cache.config.num_denoise_calls != sensitivity_field.values.shape[1]
            or cache.config.num_streams != sensitivity_field.values.shape[2]
        ):
            raise ValueError("Grouped cache configuration does not match the sensitivity topology.")
        for layer in group:
            module_name = sensitivity_field.module_names[layer]
            linear = _resolve_linear(model, module_name)
            if progress is not None:
                progress(layer, total, module_name)
            d_result = calibrate_wam_d_for_layer(
                linear.weight,
                layer=layer,
                cache=cache,
                sensitivity_weights=derived,
                config=d_config,
            )
            gamma_result = calibrate_wam_gamma_for_layer(
                layer=layer,
                cache=cache,
                sensitivity_weights=derived,
                d_result=d_result,
                config=gamma_config,
            )
            entries.append(
                WAMLinearCheckpoint(
                    layer=layer,
                    module_name=module_name,
                    qweight=d_result.quantized_weight.qweight.detach().cpu(),
                    weight_scales=d_result.quantized_weight.scales.detach().cpu(),
                    input_scale=d_result.input_scale.detach().cpu(),
                    stream_gains=gamma_result.stream_gains.detach().cpu(),
                    clipping_ranges=gamma_result.clipping_ranges.detach().cpu(),
                    token_fractions=gamma_result.token_fractions.detach().cpu(),
                    stream_indices=gamma_result.stream_indices,
                    d_initial_hard_loss=d_result.initial_hard_loss,
                    d_final_hard_loss=d_result.final_hard_loss,
                    gamma_initial_hard_losses=gamma_result.initial_hard_losses,
                    gamma_final_hard_losses=gamma_result.final_hard_losses,
                    input_rotation_signs=(
                        None
                        if input_rotation_signs is None
                        else input_rotation_signs[module_name].detach().float().cpu()
                    ),
                )
            )
            if layer_completed is not None:
                layer_completed(entries[-1])
            cache.release_layer(layer)
            del d_result, gamma_result
        del cache
    entries.sort(key=lambda entry: entry.layer)
    checkpoint = WAMCalibrationCheckpoint(
        weight_bits=d_config.weight_bits,
        activation_bits=d_config.activation_bits,
        num_denoise_calls=sensitivity_field.values.shape[1],
        sensitivity=sensitivity_field.values.detach().cpu(),
        active_stream_mask=sensitivity_field.active_stream_mask.detach().cpu(),
        timestep_weights=derived.timestep_weights.detach().cpu(),
        rho=float(rho),
        eps=float(eps),
        u_min=u_min,
        u_max=u_max,
        d_config=asdict(d_config),
        gamma_config=asdict(gamma_config),
        linears=tuple(entries),
    )
    checkpoint.validate()
    return checkpoint


def replace_linears_with_wam(
    model: nn.Module,
    checkpoint: WAMCalibrationCheckpoint,
    *,
    keep_output_dtype: str = "input",
    backend=None,
) -> int:
    """Replace every calibrated source Linear with its WAM runtime module."""

    checkpoint.validate()
    resolved: list[tuple[nn.Module, str, nn.Linear, WAMLinearCheckpoint]] = []
    for entry in checkpoint.linears:
        actual_name, child = _resolve_named_linear(model, entry.module_name)
        parent_name, _, child_name = actual_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if child.weight.shape != entry.qweight.shape:
            raise ValueError(f"Checkpoint shape does not match {entry.module_name!r}.")
        resolved.append((parent, child_name, child, entry))

    for parent, child_name, linear, entry in resolved:
        replacement = WAMFakeQuantLinear(
            qweight=entry.qweight,
            scales=entry.weight_scales,
            bias=linear.bias,
            input_scale=entry.input_scale,
            stream_gains=entry.stream_gains,
            clipping_ranges=entry.clipping_ranges,
            token_fractions=entry.token_fractions,
            weight_bits=checkpoint.weight_bits,
            activation_bits=checkpoint.activation_bits,
            input_rotation_signs=entry.input_rotation_signs,
            keep_output_dtype=keep_output_dtype,
            backend=backend,
        )
        replacement.to(device=linear.weight.device)
        replacement.train(linear.training)
        setattr(parent, child_name, replacement)
    setattr(model, "_wam_fake_quant_linear_count", len(resolved))
    bind_wam_timestep_state(model)
    return len(resolved)


class WAMNetworkCallTracker:
    """Assign successive root-network forwards to WAM denoise-call lookups."""

    def __init__(self, model: nn.Module, num_denoise_calls: int) -> None:
        if int(num_denoise_calls) <= 0:
            raise ValueError("num_denoise_calls must be positive.")
        self.model = model
        self.num_denoise_calls = int(num_denoise_calls)
        self.total_calls = 0
        self.current_call = -1
        self._handle: Any | None = None

    def _pre_hook(self, _module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        self.current_call = self.total_calls % self.num_denoise_calls
        set_wam_timestep(self.model, self.current_call)
        self.total_calls += 1

    def register(self) -> None:
        if self._handle is not None:
            raise RuntimeError("WAM network-call tracker is already registered.")
        self._handle = self.model.register_forward_pre_hook(self._pre_hook)

    def reset(self) -> None:
        self.total_calls = 0
        self.current_call = -1

    def validate_complete_schedule(self) -> None:
        if self.total_calls == 0 or self.total_calls % self.num_denoise_calls:
            raise RuntimeError(
                f"Observed {self.total_calls} network calls, which is not a complete "
                f"{self.num_denoise_calls}-call WAM schedule."
            )

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


__all__ = [
    "WAMCalibrationCheckpoint",
    "WAMLinearCheckpoint",
    "WAMNetworkCallTracker",
    "WAMSensitivityField",
    "build_wam_cache_sites",
    "calibrate_wam_model",
    "calibrate_wam_model_grouped",
    "load_wam_sensitivity_field_csv",
    "replace_linears_with_wam",
]
