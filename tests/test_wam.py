from __future__ import annotations

import pytest
import torch

from steerquant.algorithms.wam import build_wam_sensitivity_weights


def test_wam_sensitivity_weights_follow_method_normalizations() -> None:
    sensitivity = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 4.0],
            ],
            [
                [3.0, 1.0, 2.0],
                [1.0, 5.0, 2.0],
            ],
        ]
    )
    pi = torch.tensor([0.25, 0.75])

    result = build_wam_sensitivity_weights(
        sensitivity,
        timestep_weights=pi,
        rho=1.0,
        eps=1e-6,
    )

    expected_energy = sensitivity.square()
    expected_u = expected_energy + 1e-6
    expected_z_d = (pi[None, :, None] * expected_u).sum(dim=(1, 2), keepdim=True)

    torch.testing.assert_close(result.energy, expected_energy)
    torch.testing.assert_close(result.temperature_weights, expected_u)
    torch.testing.assert_close(result.omega_d, expected_u / expected_z_d)
    torch.testing.assert_close(result.omega_gamma, expected_u / expected_u.sum(dim=2, keepdim=True))
    torch.testing.assert_close(result.d_loss_weights.sum(dim=(1, 2)), torch.ones(2))
    torch.testing.assert_close(result.omega_gamma.sum(dim=2), torch.ones(2, 2))


def test_wam_rho_below_one_flattens_sensitivity_contrast() -> None:
    sensitivity = torch.tensor([[[1.0, 10.0]]])

    linear = build_wam_sensitivity_weights(sensitivity, rho=1.0)
    flattened = build_wam_sensitivity_weights(sensitivity, rho=0.5)

    linear_ratio = linear.temperature_weights[0, 0, 1] / linear.temperature_weights[0, 0, 0]
    flattened_ratio = flattened.temperature_weights[0, 0, 1] / flattened.temperature_weights[0, 0, 0]
    assert flattened_ratio < linear_ratio


def test_wam_active_stream_mask_supports_global_context_layers() -> None:
    sensitivity = torch.ones(2, 2, 3)
    mask = torch.tensor(
        [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 0, 0], [1, 0, 0]],
        ],
        dtype=torch.bool,
    )

    result = build_wam_sensitivity_weights(sensitivity, active_stream_mask=mask)

    torch.testing.assert_close(result.omega_gamma[1, :, 0], torch.ones(2))
    torch.testing.assert_close(result.omega_gamma[1, :, 1:], torch.zeros(2, 2))
    assert torch.equal(result.active_stream_mask, mask)


def test_wam_importance_clipping_is_applied_before_normalization() -> None:
    result = build_wam_sensitivity_weights(
        torch.tensor([[[0.0, 2.0, 100.0]]]),
        rho=1.0,
        eps=1e-6,
        u_min=0.25,
        u_max=4.0,
    )

    torch.testing.assert_close(result.temperature_weights, torch.tensor([[[0.25, 4.0, 4.0]]]))
    torch.testing.assert_close(result.omega_gamma.sum(dim=2), torch.ones(1, 1))


@pytest.mark.parametrize(
    ("sensitivity", "message"),
    [
        (torch.tensor([[1.0, 2.0]]), "shape"),
        (torch.tensor([[[-1.0, 2.0]]]), "non-negative"),
        (torch.tensor([[[float("nan"), 2.0]]]), "finite"),
    ],
)
def test_wam_rejects_invalid_sensitivity_fields(
    sensitivity: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_wam_sensitivity_weights(sensitivity)


def test_wam_rejects_invalid_hyperparameters() -> None:
    sensitivity = torch.ones(2, 2, 3)

    with pytest.raises(ValueError, match="rho"):
        build_wam_sensitivity_weights(sensitivity, rho=0.0)
    with pytest.raises(ValueError, match="eps"):
        build_wam_sensitivity_weights(sensitivity, eps=0.0)
    with pytest.raises(ValueError, match="Expected 2 timestep weights"):
        build_wam_sensitivity_weights(sensitivity, timestep_weights=torch.ones(3))
    with pytest.raises(ValueError, match="must not exceed"):
        build_wam_sensitivity_weights(sensitivity, u_min=2.0, u_max=1.0)
