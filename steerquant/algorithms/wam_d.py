"""Static channel-wise WAM ``D`` calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..wam_cache import WAMActivationCache, WAMCacheKey
from .quantization import QuantizedLinearWeight, quantize_weight_rtn
from .wam import WAMSensitivityWeights


@dataclass(frozen=True)
class WAMDCalibrationConfig:
    weight_bits: int = 8
    activation_bits: int = 8
    epochs: int = 20
    learning_rate: float = 2e-2
    batch_size: int = 128
    smoothquant_alpha: float = 0.5
    log_scale_bound: float = 6.0
    hard_eval_interval: int = 1
    require_all_tokens: bool = False
    train_rows_per_cell: int | None = None
    sampling_seed: int = 20260831

    def validate(self) -> None:
        for name in ("weight_bits", "activation_bits"):
            bits = int(getattr(self, name))
            if not 2 <= bits <= 8:
                raise ValueError(f"{name} must be in [2, 8], got {bits}.")
        if int(self.epochs) < 0:
            raise ValueError("epochs must be non-negative.")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0.0 <= float(self.smoothquant_alpha) <= 1.0:
            raise ValueError("smoothquant_alpha must be in [0, 1].")
        if float(self.log_scale_bound) <= 0:
            raise ValueError("log_scale_bound must be positive.")
        if int(self.hard_eval_interval) <= 0:
            raise ValueError("hard_eval_interval must be positive.")
        if self.train_rows_per_cell is not None and int(self.train_rows_per_cell) <= 0:
            raise ValueError("train_rows_per_cell must be positive or None.")


@dataclass(frozen=True)
class WAMDCalibrationResult:
    layer: int
    module_name: str
    input_scale: torch.Tensor
    quantized_weight: QuantizedLinearWeight
    initial_hard_loss: float
    final_hard_loss: float
    hard_loss_history: tuple[float, ...]
    active_cells: tuple[WAMCacheKey, ...]


@dataclass(frozen=True)
class _CalibrationCell:
    key: WAMCacheKey
    values: torch.Tensor
    full_channel_absmax: torch.Tensor
    coefficient: float


def _sample_training_rows(
    values: torch.Tensor,
    *,
    row_cap: int | None,
    seed: int,
    epoch: int,
    key: WAMCacheKey,
) -> torch.Tensor:
    """Return a deterministic uniform row sample for one epoch and cache cell."""

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


def _symmetric_fake_quant(
    values: torch.Tensor,
    *,
    bits: int,
    scale: torch.Tensor,
    ste: bool,
) -> torch.Tensor:
    qmax = 2 ** (int(bits) - 1) - 1
    qmin = -qmax
    normalized = values / scale.clamp_min(1e-8)
    if ste:
        rounded = normalized + (torch.round(normalized) - normalized).detach()
    else:
        rounded = torch.round(normalized)
    return rounded.clamp(qmin, qmax) * scale


def _fake_quant_weight_per_output_channel(
    weight: torch.Tensor,
    *,
    bits: int,
    ste: bool,
) -> torch.Tensor:
    qmax = 2 ** (int(bits) - 1) - 1
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return _symmetric_fake_quant(weight, bits=bits, scale=scale, ste=ste)


def _project_log_scale_(log_scale: torch.Tensor, bound: float) -> None:
    """Fix the scale gauge and keep channel transformations numerically safe."""

    with torch.no_grad():
        log_scale.sub_(log_scale.mean())
        log_scale.clamp_(-float(bound), float(bound))


def _collect_layer_cells(
    cache: WAMActivationCache,
    weights: WAMSensitivityWeights,
    *,
    layer: int,
    in_features: int,
    require_all_tokens: bool,
) -> tuple[list[_CalibrationCell], str]:
    if not 0 <= int(layer) < weights.d_loss_weights.shape[0]:
        raise IndexError(f"layer {layer} is outside the sensitivity field.")
    if cache.config.num_denoise_calls != weights.d_loss_weights.shape[1]:
        raise ValueError("Cache and sensitivity field have different denoise-call counts.")
    if cache.config.num_streams != weights.d_loss_weights.shape[2]:
        raise ValueError("Cache and sensitivity field have different stream counts.")

    cells: list[_CalibrationCell] = []
    module_names: set[str] = set()
    for key in cache.keys:
        if key.layer != int(layer):
            continue
        values = cache.activations(key.layer, key.timestep, key.stream)
        if values.shape[1] != in_features:
            raise ValueError(f"Cache cell {key} has {values.shape[1]} channels, expected {in_features}.")
        seen_tokens = cache.num_seen_tokens(key.layer, key.timestep, key.stream)
        if require_all_tokens and values.shape[0] != seen_tokens:
            raise ValueError(
                f"Static D calibration requires all tokens, but {key} caches {values.shape[0]} of {seen_tokens}."
            )
        module_names.add(cache.module_name(key.layer, key.timestep, key.stream))
        coefficient = float(weights.d_loss_weights[key.layer, key.timestep, key.stream].item())
        if coefficient > 0:
            cells.append(
                _CalibrationCell(
                    key=key,
                    values=values,
                    full_channel_absmax=cache.channel_absmax(key.layer, key.timestep, key.stream),
                    coefficient=coefficient,
                )
            )

    if not cells:
        raise ValueError(f"No cached activation cells found for layer {layer}.")
    if len(module_names) != 1:
        raise ValueError(f"Layer {layer} cache refers to multiple modules: {sorted(module_names)}")
    coefficient_mass = sum(cell.coefficient for cell in cells)
    if abs(coefficient_mass - 1.0) > 1e-4:
        raise ValueError(
            f"Cached cells cover D-loss coefficient mass {coefficient_mass:.6f}, expected 1. "
            "Build sensitivity weights over the same active streams as the cache."
        )
    return cells, next(iter(module_names))


def _activation_channel_absmax(
    cells: list[_CalibrationCell],
    *,
    num_denoise_calls: int,
    in_features: int,
) -> torch.Tensor:
    result = torch.zeros(num_denoise_calls, in_features, dtype=torch.float32)
    for cell in cells:
        current = cell.full_channel_absmax
        result[cell.key.timestep] = torch.maximum(result[cell.key.timestep], current)
    return result.clamp_min(1e-8)


def _initial_input_scale(
    weight: torch.Tensor,
    cells: list[_CalibrationCell],
    *,
    alpha: float,
) -> torch.Tensor:
    activation_salience = torch.zeros(weight.shape[1], dtype=torch.float32)
    for cell in cells:
        activation_salience.add_(
            cell.full_channel_absmax,
            alpha=cell.coefficient,
        )
    activation_salience.clamp_min_(1e-8)
    weight_salience = weight.detach().float().abs().amax(dim=0).cpu().clamp_min(1e-8)
    scale = activation_salience.pow(float(alpha)) / weight_salience.pow(1.0 - float(alpha))
    log_scale = scale.clamp_min(1e-8).log()
    log_scale.sub_(log_scale.mean())
    return log_scale.exp()


def _activation_scales_by_timestep(
    channel_absmax: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    bits: int,
) -> torch.Tensor:
    qmax = 2 ** (int(bits) - 1) - 1
    transformed_absmax = channel_absmax.to(input_scale.device) / input_scale[None, :]
    return transformed_absmax.amax(dim=1).clamp_min(1e-8) / qmax


def _hard_reconstruction_loss(
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    cells: list[_CalibrationCell],
    channel_absmax: torch.Tensor,
    config: WAMDCalibrationConfig,
) -> float:
    device = weight.device
    weight_fp32 = weight.detach().float()
    d = input_scale.to(device=device, dtype=torch.float32)
    transformed_weight = weight_fp32 * d[None, :]
    quantized_weight = _fake_quant_weight_per_output_channel(
        transformed_weight,
        bits=config.weight_bits,
        ste=False,
    )
    activation_scales = _activation_scales_by_timestep(
        channel_absmax,
        d,
        bits=config.activation_bits,
    )
    total = torch.zeros((), device=device, dtype=torch.float64)
    with torch.no_grad():
        for cell in cells:
            num_tokens = cell.values.shape[0]
            for start in range(0, num_tokens, config.batch_size):
                x = cell.values[start : start + config.batch_size].to(device=device, dtype=torch.float32)
                target = x @ weight_fp32.T
                transformed = x / d
                quantized_x = _symmetric_fake_quant(
                    transformed,
                    bits=config.activation_bits,
                    scale=activation_scales[cell.key.timestep],
                    ste=False,
                )
                residual = quantized_x @ quantized_weight.T - target
                total.add_(residual.double().square().sum() * (cell.coefficient / num_tokens))
    return float(total.item())


def calibrate_wam_d_for_layer(
    weight: torch.Tensor,
    *,
    layer: int,
    cache: WAMActivationCache,
    sensitivity_weights: WAMSensitivityWeights,
    config: WAMDCalibrationConfig | None = None,
) -> WAMDCalibrationResult:
    """Calibrate one positive channel-wise ``D_l`` and fold it into ``W_l^Q``."""

    config = WAMDCalibrationConfig() if config is None else config
    config.validate()
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D Linear weight, got {tuple(weight.shape)}.")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values.")

    cells, module_name = _collect_layer_cells(
        cache,
        sensitivity_weights,
        layer=int(layer),
        in_features=weight.shape[1],
        require_all_tokens=config.require_all_tokens,
    )
    channel_absmax = _activation_channel_absmax(
        cells,
        num_denoise_calls=cache.config.num_denoise_calls,
        in_features=weight.shape[1],
    )
    initial_scale = _initial_input_scale(
        weight,
        cells,
        alpha=config.smoothquant_alpha,
    ).to(weight.device)
    log_scale = torch.nn.Parameter(initial_scale.log())
    _project_log_scale_(log_scale, config.log_scale_bound)
    optimizer = torch.optim.Adam([log_scale], lr=config.learning_rate)

    initial_hard_loss = _hard_reconstruction_loss(
        weight,
        log_scale.detach().exp(),
        cells,
        channel_absmax,
        config,
    )
    best_loss = initial_hard_loss
    best_scale = log_scale.detach().exp().clone()
    history = [initial_hard_loss]
    weight_fp32 = weight.detach().float()

    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        for cell in cells:
            training_values = _sample_training_rows(
                cell.values,
                row_cap=config.train_rows_per_cell,
                seed=config.sampling_seed,
                epoch=epoch,
                key=cell.key,
            )
            num_tokens = training_values.shape[0]
            for start in range(0, num_tokens, config.batch_size):
                d = log_scale.exp()
                transformed_weight = weight_fp32 * d[None, :]
                quantized_weight = _fake_quant_weight_per_output_channel(
                    transformed_weight,
                    bits=config.weight_bits,
                    ste=True,
                )
                activation_scales = _activation_scales_by_timestep(
                    channel_absmax,
                    d,
                    bits=config.activation_bits,
                )
                x = training_values[start : start + config.batch_size].to(
                    device=weight.device,
                    dtype=torch.float32,
                )
                target = x @ weight_fp32.T
                quantized_x = _symmetric_fake_quant(
                    x / d,
                    bits=config.activation_bits,
                    scale=activation_scales[cell.key.timestep],
                    ste=True,
                )
                residual = quantized_x @ quantized_weight.T - target
                chunk_loss = residual.square().sum() * (cell.coefficient / num_tokens)
                chunk_loss.backward()
        optimizer.step()
        _project_log_scale_(log_scale, config.log_scale_bound)

        should_evaluate = (epoch + 1) % config.hard_eval_interval == 0 or epoch + 1 == config.epochs
        if should_evaluate:
            hard_loss = _hard_reconstruction_loss(
                weight,
                log_scale.detach().exp(),
                cells,
                channel_absmax,
                config,
            )
            history.append(hard_loss)
            if hard_loss < best_loss:
                best_loss = hard_loss
                best_scale = log_scale.detach().exp().clone()

    transformed_weight = weight.detach().float() * best_scale[None, :]
    qweight, scales, zeros = quantize_weight_rtn(
        transformed_weight,
        bits=config.weight_bits,
        group_size=-1,
        symmetric=True,
        granularity="per_channel",
    )
    quantized = QuantizedLinearWeight(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        input_scale=best_scale.detach().float(),
    )
    return WAMDCalibrationResult(
        layer=int(layer),
        module_name=module_name,
        input_scale=best_scale.detach().float(),
        quantized_weight=quantized,
        initial_hard_loss=initial_hard_loss,
        final_hard_loss=best_loss,
        hard_loss_history=tuple(history),
        active_cells=tuple(cell.key for cell in cells),
    )


__all__ = [
    "WAMDCalibrationConfig",
    "WAMDCalibrationResult",
    "calibrate_wam_d_for_layer",
]
