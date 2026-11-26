"""Timestep-conditioned WAM stream-gain and clipping calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..wam_cache import WAMActivationCache, WAMCacheKey
from .quantization import dequantize_weight
from .wam import WAMSensitivityWeights
from .wam_d import WAMDCalibrationResult


@dataclass(frozen=True)
class WAMGammaCalibrationConfig:
    activation_bits: int = 8
    epochs: int = 30
    gain_learning_rate: float = 2e-2
    clip_learning_rate: float = 1e-2
    batch_size: int = 128
    gamma_min: float = 0.25
    gamma_max: float = 4.0
    clip_min_ratio: float = 1e-2
    clip_max_ratio: float = 4.0
    raw_gain_bound: float = 20.0
    hard_eval_interval: int = 1
    train_rows_per_cell: int | None = None
    sampling_seed: int = 20260831

    def validate(self) -> None:
        if not 2 <= int(self.activation_bits) <= 8:
            raise ValueError("activation_bits must be in [2, 8].")
        if int(self.epochs) < 0:
            raise ValueError("epochs must be non-negative.")
        if float(self.gain_learning_rate) <= 0 or float(self.clip_learning_rate) <= 0:
            raise ValueError("gain and clip learning rates must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0 < float(self.gamma_min) < 1 < float(self.gamma_max):
            raise ValueError("gamma bounds must satisfy 0 < gamma_min < 1 < gamma_max.")
        if not 0 < float(self.clip_min_ratio) <= 1 <= float(self.clip_max_ratio):
            raise ValueError("clip ratios must satisfy 0 < clip_min_ratio <= 1 <= clip_max_ratio.")
        if float(self.raw_gain_bound) <= 0:
            raise ValueError("raw_gain_bound must be positive.")
        if int(self.hard_eval_interval) <= 0:
            raise ValueError("hard_eval_interval must be positive.")
        if self.train_rows_per_cell is not None and int(self.train_rows_per_cell) <= 0:
            raise ValueError("train_rows_per_cell must be positive or None.")


@dataclass(frozen=True)
class WAMGammaCalibrationResult:
    layer: int
    module_name: str
    stream_gains: torch.Tensor
    clipping_ranges: torch.Tensor
    activation_scales: torch.Tensor
    token_fractions: torch.Tensor
    initial_hard_losses: tuple[float, ...]
    final_hard_losses: tuple[float, ...]
    hard_loss_histories: tuple[tuple[float, ...], ...]
    stream_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _GammaCell:
    key: WAMCacheKey
    gain_index: int
    values: torch.Tensor
    importance: float


def _sample_training_rows(
    values: torch.Tensor,
    *,
    row_cap: int | None,
    seed: int,
    epoch: int,
    key: WAMCacheKey,
) -> torch.Tensor:
    """Sample one cell uniformly for stochastic calibration training."""

    if row_cap is None or int(row_cap) >= values.shape[0]:
        return values
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int(seed)
        + int(epoch) * 1_000_003
        + int(key.layer) * 10_007
        + int(key.timestep) * 101
        + int(key.stream)
    )
    indices = torch.randperm(values.shape[0], generator=generator)[: int(row_cap)]
    return values.index_select(0, indices)


def constrained_stream_gains(
    raw_log_gains: torch.Tensor,
    token_fractions: torch.Tensor,
    *,
    gamma_min: float,
    gamma_max: float,
) -> torch.Tensor:
    """Map free parameters to bounded gains with exact weighted-log conservation.

    Weighted centering enforces ``sum_s p_s log(gamma_s) = 0``.  If a centered
    vector exceeds either log bound, one common shrink factor brings it into
    the box without changing the conservation equality.
    """

    if raw_log_gains.ndim != 1:
        raise ValueError("raw_log_gains must be one-dimensional.")
    fractions = token_fractions.to(
        device=raw_log_gains.device,
        dtype=raw_log_gains.dtype,
    )
    if fractions.shape != raw_log_gains.shape:
        raise ValueError("token_fractions must have the same shape as raw_log_gains.")
    if not torch.isfinite(fractions).all() or (fractions < 0).any() or fractions.sum() <= 0:
        raise ValueError("token_fractions must be finite, non-negative, and sum to a positive value.")
    if not 0 < float(gamma_min) < 1 < float(gamma_max):
        raise ValueError("gamma bounds must satisfy 0 < gamma_min < 1 < gamma_max.")

    fractions = fractions / fractions.sum()
    centered = raw_log_gains - (fractions * raw_log_gains).sum()
    dtype_eps = torch.finfo(centered.dtype).eps
    positive_extent = centered.max().clamp_min(dtype_eps)
    negative_extent = (-centered.min()).clamp_min(dtype_eps)
    one = centered.new_ones(())
    shrink = torch.minimum(
        one,
        torch.minimum(
            centered.new_tensor(math.log(float(gamma_max))) / positive_extent,
            centered.new_tensor(-math.log(float(gamma_min))) / negative_extent,
        ),
    )
    return (centered * shrink).exp()


def _symmetric_activation_fake_quant(
    values: torch.Tensor,
    *,
    bits: int,
    clipping_range: torch.Tensor,
    ste: bool,
) -> torch.Tensor:
    qmax = 2 ** (int(bits) - 1) - 1
    qmin = -qmax
    scale = clipping_range.clamp_min(1e-8) / qmax
    normalized = values / scale
    if ste:
        rounded = normalized + (torch.round(normalized) - normalized).detach()
    else:
        rounded = torch.round(normalized)
    return rounded.clamp(qmin, qmax) * scale


def _collect_timestep_cells(
    cache: WAMActivationCache,
    sensitivity_weights: WAMSensitivityWeights,
    *,
    layer: int,
    timestep: int,
    in_features: int,
) -> tuple[list[_GammaCell], torch.Tensor, str, tuple[int, ...]]:
    expected_shape = sensitivity_weights.omega_gamma.shape
    if not 0 <= int(layer) < expected_shape[0]:
        raise IndexError(f"layer {layer} is outside the sensitivity field.")
    if not 0 <= int(timestep) < expected_shape[1]:
        raise IndexError(f"timestep {timestep} is outside the sensitivity field.")
    if cache.config.num_denoise_calls != expected_shape[1]:
        raise ValueError("Cache and sensitivity field have different denoise-call counts.")
    if cache.config.num_streams != expected_shape[2]:
        raise ValueError("Cache and sensitivity field have different stream counts.")

    cells: list[_GammaCell] = []
    module_names: set[str] = set()
    omega = sensitivity_weights.omega_gamma[int(layer), int(timestep)]
    active_streams = tuple(int(value) for value in torch.nonzero(omega > 0).flatten().tolist())
    for gain_index, stream in enumerate(active_streams):
        key = WAMCacheKey(int(layer), int(timestep), stream)
        try:
            values = cache.activations(key.layer, key.timestep, key.stream)
        except KeyError as error:
            raise ValueError(f"Missing activation cache cell {key} for gamma calibration.") from error
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != in_features:
            raise ValueError(
                f"Cache cell {key} must contain [tokens, {in_features}] activations, got {tuple(values.shape)}."
            )
        module_names.add(cache.module_name(key.layer, key.timestep, key.stream))
        importance = float(sensitivity_weights.omega_gamma[key.layer, key.timestep, key.stream].item())
        cells.append(
            _GammaCell(
                key=key,
                gain_index=gain_index,
                values=values,
                importance=importance,
            )
        )

    if len(module_names) != 1:
        raise ValueError(
            f"Layer {layer}, timestep {timestep} refers to multiple modules: {sorted(module_names)}"
        )
    importance_mass = sum(cell.importance for cell in cells)
    if abs(importance_mass - 1.0) > 1e-4:
        raise ValueError(f"Gamma-loss importance mass is {importance_mass:.6f}, expected 1.")
    counts = torch.tensor(
        [cache.num_seen_tokens(layer, timestep, stream) for stream in active_streams],
        dtype=torch.float32,
    )
    if (counts <= 0).any():
        raise ValueError("Every active stream must contain at least one real token.")
    return cells, counts / counts.sum(), next(iter(module_names)), active_streams


def _initial_clipping_range(
    cache: WAMActivationCache,
    cells: list[_GammaCell],
    input_scale: torch.Tensor,
) -> torch.Tensor:
    d_cpu = input_scale.detach().float().cpu()
    maximum = torch.zeros((), dtype=torch.float32)
    for cell in cells:
        absmax = cache.channel_absmax(
            cell.key.layer,
            cell.key.timestep,
            cell.key.stream,
        )
        maximum = torch.maximum(maximum, (absmax / d_cpu).amax())
    return maximum.clamp_min(1e-8)


def _hard_gamma_loss(
    cells: list[_GammaCell],
    transformed_weight: torch.Tensor,
    input_scale: torch.Tensor,
    gains: torch.Tensor,
    clipping_range: torch.Tensor,
    config: WAMGammaCalibrationConfig,
) -> float:
    device = transformed_weight.device
    total = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for cell in cells:
            stream = cell.gain_index
            num_tokens = cell.values.shape[0]
            for start in range(0, num_tokens, config.batch_size):
                x = cell.values[start : start + config.batch_size].to(
                    device=device,
                    dtype=torch.float32,
                )
                x_prime = x / input_scale
                quantized = _symmetric_activation_fake_quant(
                    gains[stream] * x_prime,
                    bits=config.activation_bits,
                    clipping_range=clipping_range,
                    ste=False,
                )
                reconstructed = quantized / gains[stream]
                residual = (reconstructed - x_prime) @ transformed_weight.T
                total.add_(residual.double().square().sum() * (cell.importance / num_tokens))
    return float(total.item())


def _project_optimization_parameters_(
    raw_log_gains: torch.Tensor,
    log_clipping_range: torch.Tensor,
    token_fractions: torch.Tensor,
    *,
    raw_gain_bound: float,
    min_log_clip: float,
    max_log_clip: float,
) -> None:
    with torch.no_grad():
        fractions = token_fractions.to(raw_log_gains.device, raw_log_gains.dtype)
        raw_log_gains.sub_((fractions * raw_log_gains).sum())
        raw_log_gains.clamp_(-float(raw_gain_bound), float(raw_gain_bound))
        log_clipping_range.clamp_(float(min_log_clip), float(max_log_clip))


def calibrate_wam_gamma_for_layer(
    *,
    layer: int,
    cache: WAMActivationCache,
    sensitivity_weights: WAMSensitivityWeights,
    d_result: WAMDCalibrationResult,
    config: WAMGammaCalibrationConfig | None = None,
) -> WAMGammaCalibrationResult:
    """Jointly calibrate ``gamma_l,tau,s`` and ``c_l,tau`` for one Linear."""

    config = WAMGammaCalibrationConfig() if config is None else config
    config.validate()
    if int(layer) != int(d_result.layer):
        raise ValueError(f"Requested layer {layer}, but D result is for layer {d_result.layer}.")

    quantized_weight = d_result.quantized_weight
    if quantized_weight.input_scale is None:
        raise ValueError("D calibration result has no folded input scale.")
    if quantized_weight.zeros is not None:
        raise ValueError("WAM D calibration expects a symmetric quantized weight.")
    device = quantized_weight.qweight.device
    input_scale = d_result.input_scale.detach().to(device=device, dtype=torch.float32)
    if input_scale.ndim != 1 or not torch.isfinite(input_scale).all() or (input_scale <= 0).any():
        raise ValueError("D input scale must be a finite positive vector.")
    transformed_weight = dequantize_weight(
        quantized_weight.qweight,
        quantized_weight.scales,
        quantized_weight.zeros,
        group_size=-1,
        symmetric=True,
        dtype=torch.float32,
    ).detach()
    if transformed_weight.shape[1] != input_scale.numel():
        raise ValueError("Quantized weight and D input scale have incompatible dimensions.")

    all_gains: list[torch.Tensor] = []
    all_clipping_ranges: list[torch.Tensor] = []
    all_token_fractions: list[torch.Tensor] = []
    initial_losses: list[float] = []
    final_losses: list[float] = []
    histories: list[tuple[float, ...]] = []
    module_names: set[str] = set()
    result_stream_indices: tuple[int, ...] | None = None

    for timestep in range(cache.config.num_denoise_calls):
        cells, token_fractions_cpu, module_name, active_streams = _collect_timestep_cells(
            cache,
            sensitivity_weights,
            layer=int(layer),
            timestep=timestep,
            in_features=input_scale.numel(),
        )
        if module_name != d_result.module_name:
            raise ValueError(
                f"D result belongs to {d_result.module_name!r}, but cache belongs to {module_name!r}."
            )
        module_names.add(module_name)
        if result_stream_indices is None:
            result_stream_indices = active_streams
        elif active_streams != result_stream_indices:
            raise ValueError(
                "A runtime lookup table requires the same active stream layout at every timestep."
            )
        token_fractions = token_fractions_cpu.to(device=device, dtype=torch.float32)
        initial_clip = _initial_clipping_range(cache, cells, input_scale).to(device)
        min_log_clip = math.log(float(initial_clip.item()) * config.clip_min_ratio)
        max_log_clip = math.log(float(initial_clip.item()) * config.clip_max_ratio)

        raw_log_gains = torch.nn.Parameter(
            torch.zeros(len(active_streams), device=device, dtype=torch.float32)
        )
        log_clipping_range = torch.nn.Parameter(initial_clip.log())
        optimizer = torch.optim.Adam(
            [
                {"params": [raw_log_gains], "lr": config.gain_learning_rate},
                {"params": [log_clipping_range], "lr": config.clip_learning_rate},
            ]
        )

        initial_gains = constrained_stream_gains(
            raw_log_gains.detach(),
            token_fractions,
            gamma_min=config.gamma_min,
            gamma_max=config.gamma_max,
        )
        initial_loss = _hard_gamma_loss(
            cells,
            transformed_weight,
            input_scale,
            initial_gains,
            initial_clip,
            config,
        )
        best_loss = initial_loss
        best_gains = initial_gains.clone()
        best_clip = initial_clip.clone()
        history = [initial_loss]

        for epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            for cell in cells:
                stream = cell.gain_index
                training_values = _sample_training_rows(
                    cell.values,
                    row_cap=config.train_rows_per_cell,
                    seed=config.sampling_seed,
                    epoch=epoch,
                    key=cell.key,
                )
                num_tokens = training_values.shape[0]
                for start in range(0, num_tokens, config.batch_size):
                    gains = constrained_stream_gains(
                        raw_log_gains,
                        token_fractions,
                        gamma_min=config.gamma_min,
                        gamma_max=config.gamma_max,
                    )
                    clipping_range = log_clipping_range.exp()
                    x = training_values[start : start + config.batch_size].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    x_prime = x / input_scale
                    quantized = _symmetric_activation_fake_quant(
                        gains[stream] * x_prime,
                        bits=config.activation_bits,
                        clipping_range=clipping_range,
                        ste=True,
                    )
                    reconstructed = quantized / gains[stream]
                    residual = (reconstructed - x_prime) @ transformed_weight.T
                    chunk_loss = residual.square().sum() * (cell.importance / num_tokens)
                    chunk_loss.backward()
            optimizer.step()
            _project_optimization_parameters_(
                raw_log_gains,
                log_clipping_range,
                token_fractions,
                raw_gain_bound=config.raw_gain_bound,
                min_log_clip=min_log_clip,
                max_log_clip=max_log_clip,
            )

            should_evaluate = (epoch + 1) % config.hard_eval_interval == 0 or epoch + 1 == config.epochs
            if should_evaluate:
                evaluated_gains = constrained_stream_gains(
                    raw_log_gains.detach(),
                    token_fractions,
                    gamma_min=config.gamma_min,
                    gamma_max=config.gamma_max,
                )
                evaluated_clip = log_clipping_range.detach().exp()
                hard_loss = _hard_gamma_loss(
                    cells,
                    transformed_weight,
                    input_scale,
                    evaluated_gains,
                    evaluated_clip,
                    config,
                )
                history.append(hard_loss)
                if hard_loss < best_loss:
                    best_loss = hard_loss
                    best_gains = evaluated_gains.clone()
                    best_clip = evaluated_clip.clone()

        all_gains.append(best_gains.cpu())
        all_clipping_ranges.append(best_clip.cpu())
        all_token_fractions.append(token_fractions_cpu)
        initial_losses.append(initial_loss)
        final_losses.append(best_loss)
        histories.append(tuple(history))

    clipping_ranges = torch.stack(all_clipping_ranges)
    qmax = 2 ** (int(config.activation_bits) - 1) - 1
    return WAMGammaCalibrationResult(
        layer=int(layer),
        module_name=next(iter(module_names)),
        stream_gains=torch.stack(all_gains),
        clipping_ranges=clipping_ranges,
        activation_scales=clipping_ranges / qmax,
        token_fractions=torch.stack(all_token_fractions),
        initial_hard_losses=tuple(initial_losses),
        final_hard_losses=tuple(final_losses),
        hard_loss_histories=tuple(histories),
        stream_indices=() if result_stream_indices is None else result_stream_indices,
    )


__all__ = [
    "WAMGammaCalibrationConfig",
    "WAMGammaCalibrationResult",
    "calibrate_wam_gamma_for_layer",
    "constrained_stream_gains",
]
