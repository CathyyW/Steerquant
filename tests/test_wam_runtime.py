from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from steerquant.algorithms.quantization import dequantize_weight, quantize_weight_rtn
from steerquant.algorithms.wam_d import WAMDCalibrationResult
from steerquant.algorithms.wam_gamma import WAMGammaCalibrationResult
from steerquant.backends.reference import fake_quant_activation
from steerquant.wam_runtime import (
    WAMFakeQuantLinear,
    bind_wam_timestep_state,
    clear_wam_timestep,
    expand_wam_stream_gains,
    set_wam_timestep,
)


def test_expand_stream_gains_supports_equal_unequal_and_explicit_layouts() -> None:
    gains = torch.tensor([0.5, 1.0, 2.0])

    equal = expand_wam_stream_gains(torch.zeros(2, 6, 4), gains)
    assert equal.shape == (1, 6, 1)
    torch.testing.assert_close(equal.flatten(), torch.tensor([0.5, 0.5, 1.0, 1.0, 2.0, 2.0]))

    unequal = expand_wam_stream_gains(
        torch.zeros(2, 6, 4),
        gains,
        stream_token_counts=(1, 2, 3),
    )
    torch.testing.assert_close(unequal.flatten(), torch.tensor([0.5, 1.0, 1.0, 2.0, 2.0, 2.0]))

    explicit_axis = expand_wam_stream_gains(
        torch.zeros(2, 3, 2, 4),
        gains,
        stream_axis=1,
    )
    assert explicit_axis.shape == (1, 3, 1, 1)

    inferred_axis = expand_wam_stream_gains(torch.zeros(2, 3, 2, 2, 4), gains)
    assert inferred_axis.shape == (1, 3, 1, 1, 1)

    stream_ids = torch.tensor([[2, 0, 1], [1, 2, 0]])
    explicit_ids = expand_wam_stream_gains(
        torch.zeros(2, 3, 4),
        gains,
        stream_ids=stream_ids,
    )
    torch.testing.assert_close(explicit_ids.squeeze(-1), gains[stream_ids])


def _runtime_fixture() -> tuple[nn.Linear, WAMDCalibrationResult, WAMGammaCalibrationResult]:
    linear = nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]]))
        linear.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    d = torch.tensor([2.0, 0.5])
    qweight, scales, zeros = quantize_weight_rtn(
        linear.weight * d[None, :],
        bits=4,
        group_size=-1,
        symmetric=True,
        granularity="per_channel",
    )
    from steerquant.algorithms.quantization import QuantizedLinearWeight

    quantized = QuantizedLinearWeight(
        qweight=qweight,
        scales=scales,
        zeros=zeros,
        input_scale=d,
    )
    d_result = WAMDCalibrationResult(
        layer=0,
        module_name="blocks.0.self_attn.q_proj",
        input_scale=d,
        quantized_weight=quantized,
        initial_hard_loss=1.0,
        final_hard_loss=0.5,
        hard_loss_history=(1.0, 0.5),
        active_cells=(),
    )
    gains = torch.tensor([[1.0, 1.0], [2.0, 0.5]])
    fractions = torch.full((2, 2), 0.5)
    clips = torch.tensor([4.0, 3.0])
    gamma_result = WAMGammaCalibrationResult(
        layer=0,
        module_name="blocks.0.self_attn.q_proj",
        stream_gains=gains,
        clipping_ranges=clips,
        activation_scales=clips / 7,
        token_fractions=fractions,
        initial_hard_losses=(1.0, 1.0),
        final_hard_losses=(0.5, 0.5),
        hard_loss_histories=((1.0, 0.5), (1.0, 0.5)),
    )
    return linear, d_result, gamma_result


def test_wam_runtime_matches_broadcast_reference_and_requires_timestep() -> None:
    linear, d_result, gamma_result = _runtime_fixture()
    module = WAMFakeQuantLinear.from_calibration(
        linear,
        d_result,
        gamma_result,
        weight_bits=4,
        activation_bits=4,
    )
    x = torch.tensor([[[1.0, -1.0], [2.0, 0.5], [-0.5, 3.0], [1.5, -2.0]]])

    with pytest.raises(RuntimeError, match="denoise-call index"):
        module(x)
    module.set_timestep(1)
    actual = module(x)

    gain_rows = torch.tensor([[[2.0], [2.0], [0.5], [0.5]]])
    x_prime = x / d_result.input_scale
    quantized_x = fake_quant_activation(
        x_prime * gain_rows,
        bits=4,
        symmetric=True,
        granularity="per_tensor",
        static_scale=torch.tensor(3.0 / 7),
    )
    weight = dequantize_weight(
        d_result.quantized_weight.qweight,
        d_result.quantized_weight.scales,
        None,
        group_size=-1,
        symmetric=True,
        dtype=torch.float32,
    )
    expected = F.linear(quantized_x / gain_rows, weight, linear.bias)
    torch.testing.assert_close(actual, expected)

    five_dimensional = module(torch.randn(1, 2, 2, 2, 2))
    assert five_dimensional.shape == (1, 2, 2, 2, 3)
    assert torch.isfinite(five_dimensional).all()


def test_wam_timestep_helpers_update_all_runtime_modules() -> None:
    linear, d_result, gamma_result = _runtime_fixture()
    model = nn.Sequential(
        WAMFakeQuantLinear.from_calibration(
            linear,
            d_result,
            gamma_result,
            weight_bits=4,
            activation_bits=4,
        ),
        nn.ReLU(),
        WAMFakeQuantLinear.from_calibration(
            linear,
            d_result,
            gamma_result,
            weight_bits=4,
            activation_bits=4,
        ),
    )

    assert set_wam_timestep(model, 1) == 2
    assert model[0].timestep == model[2].timestep == 1
    assert clear_wam_timestep(model) == 2
    assert model[0].timestep is None and model[2].timestep is None


def test_wam_timestep_helpers_use_shared_network_state() -> None:
    linear, d_result, gamma_result = _runtime_fixture()
    model = nn.Sequential(
        WAMFakeQuantLinear.from_calibration(linear, d_result, gamma_result, weight_bits=4, activation_bits=4),
        WAMFakeQuantLinear.from_calibration(linear, d_result, gamma_result, weight_bits=4, activation_bits=4),
    )

    assert bind_wam_timestep_state(model) == 2
    assert set_wam_timestep(model, 1) == 2
    assert model[0].timestep == model[1].timestep == 1
    assert clear_wam_timestep(model) == 2
    assert model[0].timestep is None and model[1].timestep is None
