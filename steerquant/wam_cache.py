"""Stream-aware activation cache for WAM offline calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

WAM_CACHE_FORMAT = "cosmos_policy_wam_activation_cache_v2"


@dataclass(frozen=True)
class WAMCacheConfig:
    num_layers: int
    num_denoise_calls: int
    num_streams: int
    max_observations: int = 320
    tokens_per_observation: int | None = 16
    seed: int = 20260827
    storage_dtype: str = "float16"

    def validate(self) -> None:
        for name in ("num_layers", "num_denoise_calls", "num_streams"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if int(self.max_observations) <= 0:
            raise ValueError("max_observations must be positive.")
        if self.tokens_per_observation is not None and int(self.tokens_per_observation) <= 0:
            raise ValueError("tokens_per_observation must be positive or None for all tokens.")
        if self.storage_dtype not in ("float16", "float32"):
            raise ValueError("storage_dtype must be 'float16' or 'float32'.")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float16 if self.storage_dtype == "float16" else torch.float32


@dataclass(frozen=True, order=True)
class WAMCacheKey:
    layer: int
    timestep: int
    stream: int


@dataclass
class _WAMCacheCell:
    module_name: str
    in_features: int
    samples: list[torch.Tensor] = field(default_factory=list)
    row_observation_ids: list[torch.Tensor] = field(default_factory=list)
    observation_ids: set[int] = field(default_factory=set)
    num_seen_tokens: int = 0
    full_channel_absmax: torch.Tensor | None = None

    def activations(self, dtype: torch.dtype) -> torch.Tensor:
        if not self.samples:
            return torch.empty(0, self.in_features, dtype=dtype)
        return torch.cat(self.samples, dim=0)

    def row_ids(self) -> torch.Tensor:
        if not self.row_observation_ids:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(self.row_observation_ids, dim=0)


def split_activation_streams(
    values: torch.Tensor,
    *,
    num_streams: int,
    slot_aligned: bool,
) -> tuple[torch.Tensor, ...]:
    """Split a Linear input into stream tensors without padding streams.

    Slot-aligned WAM tensors are accepted as ``[B, S, ..., D]`` or flattened
    ``[B, N, D]`` tensors whose token dimension is divisible by ``S``.  A
    non-slot-aligned input (for example cross-attention K/V text context) is
    returned as one global stream.
    """

    if not isinstance(values, torch.Tensor) or values.ndim < 2:
        raise ValueError("activation values must be a Tensor with a hidden dimension.")
    if int(num_streams) <= 0:
        raise ValueError("num_streams must be positive.")
    if not slot_aligned:
        return (values,)

    if values.ndim >= 4:
        if values.shape[1] != num_streams:
            raise ValueError(f"Expected stream dimension {num_streams} at axis 1, got {tuple(values.shape)}.")
        return tuple(values[:, stream] for stream in range(num_streams))

    if values.ndim == 3:
        if values.shape[1] % num_streams:
            raise ValueError(
                f"Flattened token count {values.shape[1]} is not divisible by {num_streams} streams."
            )
        grid = values.reshape(values.shape[0], num_streams, -1, values.shape[-1])
        return tuple(grid[:, stream] for stream in range(num_streams))

    raise ValueError(
        f"Slot-aligned activations must have shape [B, N, D] or [B, S, ..., D], got {tuple(values.shape)}."
    )


class WAMActivationCache:
    """CPU cache for exact or explicitly sampled WAM reconstruction calibration."""

    def __init__(self, config: WAMCacheConfig) -> None:
        config.validate()
        self.config = config
        self._cells: dict[WAMCacheKey, _WAMCacheCell] = {}
        self._accepted_observations: set[int] = set()

    @property
    def accepted_observations(self) -> tuple[int, ...]:
        return tuple(sorted(self._accepted_observations))

    @property
    def keys(self) -> tuple[WAMCacheKey, ...]:
        return tuple(sorted(self._cells))

    @property
    def storage_nbytes(self) -> int:
        """Approximate resident tensor storage without concatenating cells."""

        total = 0
        for cell in self._cells.values():
            total += sum(value.numel() * value.element_size() for value in cell.samples)
            total += sum(value.numel() * value.element_size() for value in cell.row_observation_ids)
            if cell.full_channel_absmax is not None:
                total += cell.full_channel_absmax.numel() * cell.full_channel_absmax.element_size()
        return int(total)

    def release_layer(self, layer: int) -> int:
        """Release all cached tensors for a calibrated layer and return cell count."""

        layer = int(layer)
        if not 0 <= layer < self.config.num_layers:
            raise IndexError(f"layer index {layer} is outside [0, {self.config.num_layers}).")
        selected = [key for key in self._cells if key.layer == layer]
        for key in selected:
            del self._cells[key]
        return len(selected)

    def _validate_key(self, key: WAMCacheKey) -> None:
        bounds = (
            ("layer", key.layer, self.config.num_layers),
            ("timestep", key.timestep, self.config.num_denoise_calls),
            ("stream", key.stream, self.config.num_streams),
        )
        for name, value, upper in bounds:
            if not 0 <= int(value) < int(upper):
                raise IndexError(f"{name} index {value} is outside [0, {upper}).")

    def accepts_observation(self, observation_id: int) -> bool:
        observation_id = int(observation_id)
        if observation_id in self._accepted_observations:
            return True
        if len(self._accepted_observations) >= self.config.max_observations:
            return False
        self._accepted_observations.add(observation_id)
        return True

    def add(
        self,
        *,
        layer: int,
        timestep: int,
        stream: int,
        observation_id: int,
        module_name: str,
        values: torch.Tensor,
    ) -> bool:
        """Store activation rows for one cache cell.

        Returns ``True`` when this observation contributed rows.  Replays of an
        already-recorded observation/cell and observations beyond the configured
        cap return ``False`` and do not change token counts.  All rows are kept
        when ``tokens_per_observation`` is ``None``; otherwise a deterministic
        uniform sample up to that cap is stored.  The full-token channel-wise
        absolute maximum is updated before sampling.
        """

        key = WAMCacheKey(int(layer), int(timestep), int(stream))
        self._validate_key(key)
        observation_id = int(observation_id)
        if not self.accepts_observation(observation_id):
            return False
        if not isinstance(values, torch.Tensor) or values.ndim < 2:
            raise ValueError("values must be a Tensor with a final hidden dimension.")

        flat = values.detach().reshape(-1, values.shape[-1])
        if flat.shape[0] == 0 or flat.shape[1] == 0:
            raise ValueError("values must contain at least one token and one hidden channel.")
        if not torch.isfinite(flat).all():
            raise ValueError("values must contain only finite activations.")

        cell = self._cells.get(key)
        if cell is None:
            cell = _WAMCacheCell(module_name=str(module_name), in_features=int(flat.shape[1]))
            self._cells[key] = cell
        elif cell.module_name != str(module_name) or cell.in_features != int(flat.shape[1]):
            raise ValueError(
                f"Cache cell {key} changed identity or hidden size: "
                f"({cell.module_name}, {cell.in_features}) vs ({module_name}, {flat.shape[1]})."
            )
        if observation_id in cell.observation_ids:
            return False

        observation_absmax = flat.float().abs().amax(dim=0).cpu()
        if cell.full_channel_absmax is None:
            cell.full_channel_absmax = observation_absmax
        else:
            cell.full_channel_absmax = torch.maximum(
                cell.full_channel_absmax,
                observation_absmax,
            )

        sample_cap = self.config.tokens_per_observation
        if sample_cap is None or sample_cap >= flat.shape[0]:
            count = int(flat.shape[0])
            sampled = flat.to(device="cpu", dtype=self.config.torch_dtype)
        else:
            count = int(sample_cap)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(self.config.seed)
                + observation_id * 1_000_003
                + key.layer * 10_007
                + key.timestep * 101
                + key.stream
            )
            indices = torch.randperm(flat.shape[0], generator=generator)[:count]
            sampled = flat.index_select(0, indices.to(flat.device)).to(
                device="cpu",
                dtype=self.config.torch_dtype,
            )
        cell.samples.append(sampled.contiguous())
        cell.row_observation_ids.append(torch.full((count,), observation_id, dtype=torch.long))
        cell.observation_ids.add(observation_id)
        cell.num_seen_tokens += int(flat.shape[0])
        return True

    def activations(self, layer: int, timestep: int, stream: int) -> torch.Tensor:
        key = WAMCacheKey(int(layer), int(timestep), int(stream))
        self._validate_key(key)
        if key not in self._cells:
            raise KeyError(f"No activation cache cell for {key}.")
        return self._cells[key].activations(self.config.torch_dtype)

    def row_observation_ids(self, layer: int, timestep: int, stream: int) -> torch.Tensor:
        key = WAMCacheKey(int(layer), int(timestep), int(stream))
        self._validate_key(key)
        if key not in self._cells:
            raise KeyError(f"No activation cache cell for {key}.")
        return self._cells[key].row_ids()

    def num_seen_tokens(self, layer: int, timestep: int, stream: int) -> int:
        key = WAMCacheKey(int(layer), int(timestep), int(stream))
        self._validate_key(key)
        return 0 if key not in self._cells else self._cells[key].num_seen_tokens

    def channel_absmax(self, layer: int, timestep: int, stream: int) -> torch.Tensor:
        """Return the exact per-channel absmax computed from all seen tokens."""

        key = WAMCacheKey(int(layer), int(timestep), int(stream))
        self._validate_key(key)
        if key not in self._cells or self._cells[key].full_channel_absmax is None:
            raise KeyError(f"No full-token channel absmax for {key}.")
        return self._cells[key].full_channel_absmax.clone()

    def module_name(self, layer: int, timestep: int, stream: int) -> str:
        key = WAMCacheKey(int(layer), int(timestep), int(stream))
        self._validate_key(key)
        if key not in self._cells:
            raise KeyError(f"No activation cache cell for {key}.")
        return self._cells[key].module_name

    def token_fractions(self, layer: int, timestep: int) -> torch.Tensor:
        """Return real-token fractions used by gain conservation."""

        counts = torch.tensor(
            [self.num_seen_tokens(layer, timestep, stream) for stream in range(self.config.num_streams)],
            dtype=torch.float64,
        )
        if counts.sum() <= 0:
            raise ValueError(f"No observed tokens for layer={layer}, timestep={timestep}.")
        return (counts / counts.sum()).to(torch.float32)

    def state_dict(self) -> dict[str, Any]:
        cells: dict[str, dict[str, Any]] = {}
        for key in self.keys:
            cell = self._cells[key]
            cells[f"{key.layer}:{key.timestep}:{key.stream}"] = {
                "module_name": cell.module_name,
                "in_features": cell.in_features,
                "activations": cell.activations(self.config.torch_dtype),
                "row_observation_ids": cell.row_ids(),
                "observation_ids": sorted(cell.observation_ids),
                "num_seen_tokens": cell.num_seen_tokens,
                "full_channel_absmax": cell.full_channel_absmax,
            }
        return {
            "format": WAM_CACHE_FORMAT,
            "config": asdict(self.config),
            "accepted_observations": list(self.accepted_observations),
            "cells": cells,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> WAMActivationCache:
        if not isinstance(payload, dict) or payload.get("format") != WAM_CACHE_FORMAT:
            raise ValueError(f"Expected activation cache format {WAM_CACHE_FORMAT!r}.")
        cache = cls(WAMCacheConfig(**payload["config"]))
        cache._accepted_observations = {int(value) for value in payload["accepted_observations"]}
        for encoded_key, stored in payload["cells"].items():
            layer, timestep, stream = (int(value) for value in encoded_key.split(":"))
            key = WAMCacheKey(layer, timestep, stream)
            cache._validate_key(key)
            activations = torch.as_tensor(stored["activations"], dtype=cache.config.torch_dtype).cpu()
            row_ids = torch.as_tensor(stored["row_observation_ids"], dtype=torch.long).cpu()
            if activations.ndim != 2 or row_ids.shape != (activations.shape[0],):
                raise ValueError(f"Malformed activation cache cell {encoded_key}.")
            full_channel_absmax = torch.as_tensor(stored["full_channel_absmax"], dtype=torch.float32).cpu()
            if full_channel_absmax.shape != (int(stored["in_features"]),):
                raise ValueError(f"Malformed channel absmax for cache cell {encoded_key}.")
            cache._cells[key] = _WAMCacheCell(
                module_name=str(stored["module_name"]),
                in_features=int(stored["in_features"]),
                samples=[activations],
                row_observation_ids=[row_ids],
                observation_ids={int(value) for value in stored["observation_ids"]},
                num_seen_tokens=int(stored["num_seen_tokens"]),
                full_channel_absmax=full_channel_absmax,
            )
        return cache

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), resolved)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> WAMActivationCache:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(resolved, map_location="cpu")
        return cls.from_state_dict(payload)


@dataclass(frozen=True)
class WAMCacheSite:
    layer: int
    module_name: str
    module: nn.Module
    slot_aligned: bool
    stream_indices: tuple[int, ...]


class WAMActivationCacheCollector:
    """Hook selected Linear inputs into a :class:`WAMActivationCache`."""

    def __init__(self, cache: WAMActivationCache, sites: list[WAMCacheSite]) -> None:
        self.cache = cache
        self.sites = sites
        self._handles: list[Any] = []
        self._observation_id: int | None = None
        self._timestep: int | None = None
        self._active = False

    def begin_observation(self, observation_id: int) -> bool:
        self._observation_id = int(observation_id)
        self._timestep = None
        self._active = self.cache.accepts_observation(observation_id)
        return self._active

    def begin_denoise_call(self, timestep: int, _sigma: torch.Tensor | None = None) -> None:
        if not 0 <= int(timestep) < self.cache.config.num_denoise_calls:
            raise IndexError(f"timestep {timestep} is outside the configured denoise schedule.")
        self._timestep = int(timestep)

    def _capture(self, site: WAMCacheSite, inputs: tuple[Any, ...]) -> None:
        if not self._active:
            return
        if self._observation_id is None or self._timestep is None:
            raise RuntimeError("Call begin_observation and begin_denoise_call before model forward.")
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError(f"{site.module_name} did not receive a Tensor as its first input.")
        streams = split_activation_streams(
            inputs[0],
            num_streams=len(site.stream_indices),
            slot_aligned=site.slot_aligned,
        )
        if len(streams) != len(site.stream_indices):
            raise RuntimeError(f"Stream mapping mismatch for {site.module_name}.")
        for stream_index, values in zip(site.stream_indices, streams):
            self.cache.add(
                layer=site.layer,
                timestep=self._timestep,
                stream=stream_index,
                observation_id=self._observation_id,
                module_name=site.module_name,
                values=values,
            )

    def register(self) -> None:
        if self._handles:
            raise RuntimeError("WAM activation cache collector is already registered.")
        for site in self.sites:
            self._handles.append(
                site.module.register_forward_pre_hook(
                    lambda _module, inputs, selected=site: self._capture(selected, inputs)
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = [
    "WAMActivationCache",
    "WAMActivationCacheCollector",
    "WAMCacheConfig",
    "WAMCacheKey",
    "WAMCacheSite",
    "split_activation_streams",
]
