from __future__ import annotations

import torch

from steerquant.algorithms.wam import build_wam_sensitivity_weights
from steerquant.algorithms.wam_d import (
    WAMDCalibrationConfig,
    calibrate_wam_d_for_layer,
)
from steerquant.algorithms.wam_gamma import (
    WAMGammaCalibrationConfig,
    _sample_training_rows,
    calibrate_wam_gamma_for_layer,
    constrained_stream_gains,
)
from steerquant.wam_cache import (
    WAMActivationCache,
    WAMCacheConfig,
    WAMCacheKey,
)


def _cache() -> WAMActivationCache:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=1,
            num_denoise_calls=2,
            num_streams=3,
            max_observations=2,
            tokens_per_observation=2,
            storage_dtype="float32",
        )
    )
    generator = torch.Generator().manual_seed(31)
    for observation in range(2):
        for timestep in range(2):
            for stream in range(3):
                values = torch.randn(3 + stream, 4, generator=generator)
                values[:, stream] *= 3.0 + 2.0 * timestep
                cache.add(
                    layer=0,
                    timestep=timestep,
                    stream=stream,
                    observation_id=observation,
                    module_name="blocks.0.self_attn.q_proj",
                    values=values,
                )
    return cache


def test_constrained_stream_gains_satisfy_bounds_and_weighted_conservation() -> None:
    raw = torch.tensor([5.0, -3.0, 1.0])
    fractions = torch.tensor([0.6, 0.3, 0.1])

    gains = constrained_stream_gains(
        raw,
        fractions,
        gamma_min=0.5,
        gamma_max=3.0,
    )

    assert (gains >= 0.5 - 1e-6).all()
    assert (gains <= 3.0 + 1e-6).all()
    torch.testing.assert_close(
        (fractions * gains.log()).sum(),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_gamma_calibration_uses_frozen_d_and_weight_result() -> None:
    torch.manual_seed(17)
    cache = _cache()
    weight = torch.randn(5, 4)
    sensitivity_weights = build_wam_sensitivity_weights(
        torch.tensor([[[1.0, 3.0, 6.0], [5.0, 2.0, 1.0]]]),
        rho=0.5,
    )
    d_result = calibrate_wam_d_for_layer(
        weight,
        layer=0,
        cache=cache,
        sensitivity_weights=sensitivity_weights,
        config=WAMDCalibrationConfig(
            weight_bits=4,
            activation_bits=4,
            epochs=1,
            batch_size=4,
        ),
    )
    input_scale_before = d_result.input_scale.clone()
    qweight_before = d_result.quantized_weight.qweight.clone()

    result = calibrate_wam_gamma_for_layer(
        layer=0,
        cache=cache,
        sensitivity_weights=sensitivity_weights,
        d_result=d_result,
        config=WAMGammaCalibrationConfig(
            activation_bits=4,
            epochs=3,
            gain_learning_rate=5e-2,
            clip_learning_rate=5e-2,
            batch_size=3,
            gamma_min=0.5,
            gamma_max=2.0,
        ),
    )

    assert result.module_name == "blocks.0.self_attn.q_proj"
    assert result.stream_gains.shape == (2, 3)
    assert result.clipping_ranges.shape == (2,)
    assert result.activation_scales.shape == (2,)
    assert result.token_fractions.shape == (2, 3)
    assert torch.isfinite(result.stream_gains).all()
    assert (result.stream_gains >= 0.5 - 1e-6).all()
    assert (result.stream_gains <= 2.0 + 1e-6).all()
    assert torch.isfinite(result.clipping_ranges).all()
    assert (result.clipping_ranges > 0).all()
    torch.testing.assert_close(result.activation_scales, result.clipping_ranges / 7)
    torch.testing.assert_close(
        (result.token_fractions * result.stream_gains.log()).sum(dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )
    assert all(
        final <= initial for initial, final in zip(result.initial_hard_losses, result.final_hard_losses)
    )
    torch.testing.assert_close(d_result.input_scale, input_scale_before)
    torch.testing.assert_close(d_result.quantized_weight.qweight, qweight_before)


def test_gamma_calibration_epoch_zero_returns_identity_gains() -> None:
    cache = _cache()
    weight = torch.randn(5, 4)
    sensitivity_weights = build_wam_sensitivity_weights(torch.ones(1, 2, 3))
    d_result = calibrate_wam_d_for_layer(
        weight,
        layer=0,
        cache=cache,
        sensitivity_weights=sensitivity_weights,
        config=WAMDCalibrationConfig(epochs=0),
    )

    result = calibrate_wam_gamma_for_layer(
        layer=0,
        cache=cache,
        sensitivity_weights=sensitivity_weights,
        d_result=d_result,
        config=WAMGammaCalibrationConfig(epochs=0),
    )

    torch.testing.assert_close(result.stream_gains, torch.ones(2, 3))
    assert result.initial_hard_losses == result.final_hard_losses


def test_gamma_random_training_rows_use_full_cell_when_cap_is_not_binding() -> None:
    values = torch.randn(7, 4)
    key = WAMCacheKey(0, 1, 2)

    sampled = _sample_training_rows(values, row_cap=20, seed=9, epoch=3, key=key)

    assert sampled is values
