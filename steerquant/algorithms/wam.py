"""Core mathematics for action-sensitivity-guided WAM quantization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WAMSensitivityWeights:
    """Derived optimization weights for the full WAM sensitivity field.

    Every tensor has shape ``[num_layers, num_denoise_calls, num_streams]`` except
    ``timestep_weights``, which has shape ``[num_denoise_calls]``.
    """

    energy: torch.Tensor
    temperature_weights: torch.Tensor
    timestep_weights: torch.Tensor
    omega_d: torch.Tensor
    omega_gamma: torch.Tensor
    active_stream_mask: torch.Tensor

    @property
    def d_loss_weights(self) -> torch.Tensor:
        """Return the actual coefficients ``pi_tau * omega^D_tau,s``."""

        return self.timestep_weights[None, :, None] * self.omega_d


def _validate_clip_bounds(u_min: float | None, u_max: float | None) -> None:
    if u_min is not None and u_min < 0:
        raise ValueError(f"u_min must be non-negative, got {u_min}.")
    if u_max is not None and u_max <= 0:
        raise ValueError(f"u_max must be positive, got {u_max}.")
    if u_min is not None and u_max is not None and u_min > u_max:
        raise ValueError(f"u_min ({u_min}) must not exceed u_max ({u_max}).")


def build_wam_sensitivity_weights(
    sensitivity: torch.Tensor,
    *,
    timestep_weights: torch.Tensor | None = None,
    rho: float = 1.0,
    eps: float = 1e-8,
    u_min: float | None = None,
    u_max: float | None = None,
    active_stream_mask: torch.Tensor | None = None,
) -> WAMSensitivityWeights:
    """Convert the full ``S[layer, tau, stream]`` field into WAM weights."""

    values = torch.as_tensor(sensitivity)
    if values.ndim != 3:
        raise ValueError(
            f"sensitivity must have shape [num_layers, num_denoise_calls, num_streams], got {tuple(values.shape)}."
        )
    if values.numel() == 0 or any(size == 0 for size in values.shape):
        raise ValueError("sensitivity must contain at least one layer, denoise call, and stream.")
    if not values.is_floating_point():
        values = values.float()
    else:
        values = values.to(dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("sensitivity must contain only finite values.")
    if (values < 0).any():
        raise ValueError("sensitivity is an RMSE field and must be non-negative.")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}.")
    if float(eps) <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    _validate_clip_bounds(u_min, u_max)

    num_calls = values.shape[1]
    if timestep_weights is None:
        pi = torch.full(
            (num_calls,),
            1.0 / num_calls,
            device=values.device,
            dtype=torch.float32,
        )
    else:
        pi = torch.as_tensor(timestep_weights, device=values.device, dtype=torch.float32).flatten()
        if pi.numel() != num_calls:
            raise ValueError(f"Expected {num_calls} timestep weights, got {pi.numel()}.")
        if not torch.isfinite(pi).all() or (pi < 0).any() or pi.sum() <= 0:
            raise ValueError("timestep_weights must be finite, non-negative, and have a positive sum.")
        pi = pi / pi.sum()

    energy = values.square()
    importance = (energy + float(eps)).pow(float(rho))
    if u_min is not None or u_max is not None:
        importance = torch.clamp(importance, min=u_min, max=u_max)

    if active_stream_mask is None:
        active_mask = torch.ones_like(importance, dtype=torch.bool)
    else:
        supplied_mask = torch.as_tensor(active_stream_mask, device=values.device)
        if supplied_mask.shape != values.shape:
            raise ValueError(
                "active_stream_mask must have the same [layer, timestep, stream] "
                f"shape as sensitivity, got {tuple(supplied_mask.shape)}."
            )
        if supplied_mask.dtype != torch.bool:
            if supplied_mask.is_floating_point() and not torch.isfinite(supplied_mask).all():
                raise ValueError("active_stream_mask must contain only finite 0/1 values.")
            if not ((supplied_mask == 0) | (supplied_mask == 1)).all():
                raise ValueError("active_stream_mask must contain only boolean or 0/1 values.")
        active_mask = supplied_mask.bool()
    if not active_mask.any(dim=2).all():
        raise ValueError("Every [layer, timestep] must contain at least one active stream.")
    importance = importance * active_mask

    z_d = (pi[None, :, None] * importance).sum(dim=(1, 2), keepdim=True)
    omega_d = importance / z_d
    omega_gamma = importance / importance.sum(dim=2, keepdim=True)

    return WAMSensitivityWeights(
        energy=energy,
        temperature_weights=importance,
        timestep_weights=pi,
        omega_d=omega_d,
        omega_gamma=omega_gamma,
        active_stream_mask=active_mask,
    )
