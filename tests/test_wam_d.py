from __future__ import annotations

import pytest
import torch

from steerquant.algorithms.wam import build_wam_sensitivity_weights
from steerquant.algorithms.wam_d import (
    WAMDCalibrationConfig,
    _sample_training_rows,
    calibrate_wam_d_for_layer,
)
from steerquant.wam_cache import (
    WAMActivationCache,
    WAMCacheConfig,
    WAMCacheKey,
)


def _full_token_cache(*, sampled: bool = False) -> WAMActivationCache:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=1,
            num_denoise_calls=2,
            num_streams=2,
            max_observations=2,
            tokens_per_observation=1 if sampled else None,
            storage_dtype="float32",
        )
    )
    generator = torch.Generator().manual_seed(9)
    for observation in range(2):
        for timestep in range(2):
            for stream in range(2):
                values = torch.randn(3 + stream, 4, generator=generator)
                values[:, stream] *= 5.0 + timestep
                cache.add(
                    layer=0,
                    timestep=timestep,
                    stream=stream,
                    observation_id=observation,
                    module_name="blocks.0.self_attn.q_proj",
                    values=values,
                )
    return cache


def test_wam_d_calibration_folds_positive_scale_into_static_weight() -> None:
    torch.manual_seed(4)
    weight = torch.randn(3, 4)
    sensitivity = build_wam_sensitivity_weights(
        torch.tensor([[[1.0, 2.0], [3.0, 1.0]]]),
        rho=0.5,
    )
    result = calibrate_wam_d_for_layer(
        weight,
        layer=0,
        cache=_full_token_cache(),
        sensitivity_weights=sensitivity,
        config=WAMDCalibrationConfig(
            weight_bits=4,
            activation_bits=4,
            epochs=3,
            learning_rate=5e-2,
            batch_size=3,
        ),
    )

    assert result.module_name == "blocks.0.self_attn.q_proj"
    assert result.input_scale.shape == (4,)
    assert torch.isfinite(result.input_scale).all()
    assert (result.input_scale > 0).all()
    assert result.quantized_weight.qweight.dtype == torch.int8
    assert result.quantized_weight.qweight.shape == weight.shape
    assert result.quantized_weight.scales.shape == (3, 1)
    assert result.quantized_weight.zeros is None
    torch.testing.assert_close(result.quantized_weight.input_scale, result.input_scale)
    assert result.final_hard_loss <= result.initial_hard_loss
    assert len(result.active_cells) == 4


def test_wam_d_float_transform_is_function_preserving() -> None:
    torch.manual_seed(12)
    x = torch.randn(7, 4)
    weight = torch.randn(3, 4)
    d = torch.exp(torch.randn(4))

    reference = x @ weight.T
    transformed = (x / d) @ (weight * d[None, :]).T

    torch.testing.assert_close(transformed, reference, rtol=1e-5, atol=1e-6)


def test_wam_d_rejects_sampled_cache_in_exact_mode() -> None:
    weight = torch.randn(3, 4)
    sensitivity = build_wam_sensitivity_weights(torch.ones(1, 2, 2))

    with pytest.raises(ValueError, match="requires all tokens"):
        calibrate_wam_d_for_layer(
            weight,
            layer=0,
            cache=_full_token_cache(sampled=True),
            sensitivity_weights=sensitivity,
            config=WAMDCalibrationConfig(epochs=0, require_all_tokens=True),
        )


def test_wam_d_accepts_sampled_cache_by_default() -> None:
    weight = torch.randn(3, 4)
    sensitivity = build_wam_sensitivity_weights(torch.ones(1, 2, 2))

    result = calibrate_wam_d_for_layer(
        weight,
        layer=0,
        cache=_full_token_cache(sampled=True),
        sensitivity_weights=sensitivity,
        config=WAMDCalibrationConfig(epochs=0),
    )

    assert result.input_scale.shape == (4,)
    assert torch.isfinite(result.input_scale).all()


def test_wam_d_random_training_rows_are_balanced_deterministic_and_epoch_varying() -> None:
    values = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    key = WAMCacheKey(2, 3, 4)

    first = _sample_training_rows(values, row_cap=6, seed=17, epoch=0, key=key)
    repeated = _sample_training_rows(values, row_cap=6, seed=17, epoch=0, key=key)
    next_epoch = _sample_training_rows(values, row_cap=6, seed=17, epoch=1, key=key)

    torch.testing.assert_close(first, repeated)
    assert first.shape == (6, 4)
    assert not torch.equal(first, next_epoch)
    assert _sample_training_rows(values, row_cap=None, seed=17, epoch=0, key=key) is values
