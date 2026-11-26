from __future__ import annotations
import torch
from ..kernels.w4a8 import (
    pack_signed_int4,
    symmetric_static_linear,
    symmetric_static_stream_linear,
)
from ..kernels.w4a8.loader import load_extension
from ..kernels.w4a4.loader import load_extension as load_w4a4_extension


class CutlassW4A8Backend:
    """Packed symmetric W4A8/W4A4 inference backend for SM80+ Tensor Cores."""

    name = "cutlass"
    packs_symmetric_int4 = True

    def __init__(self) -> None:
        self._extension = None
        self._w4a4_extension = None

    def _loaded_extension(self):
        if self._extension is None:
            self._extension = load_extension()
        return self._extension

    def _loaded_w4a4_extension(self):
        if self._w4a4_extension is None:
            self._w4a4_extension = load_w4a4_extension()
        return self._w4a4_extension

    @staticmethod
    def prepare_input_scale(input_scale: torch.Tensor) -> torch.Tensor:
        """Prepare fixed WAM migration scales outside inference."""
        return input_scale.detach().float().reciprocal().contiguous()

    def wam_stream_linear_prevalidated(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
    ) -> torch.Tensor:
        """Low-overhead fused-stream WAM module hot path."""
        return self._loaded_extension().symmetric_static_stream_scheduled(
            x.contiguous(),
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
        )

    def wam_stream_w4a4_linear_prevalidated(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
    ) -> torch.Tensor:
        return self._loaded_w4a4_extension().symmetric_static_stream_scheduled(
            x.contiguous(),
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
        )

    def wam_stream_adaln_linear_prevalidated(
        self,
        x: torch.Tensor,
        adaln_scale: torch.Tensor,
        adaln_shift: torch.Tensor,
        modulation_span: int,
        epsilon: float,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
    ) -> torch.Tensor:
        """Fuse affine-free LayerNorm and AdaLN into scheduled WAM A8."""
        return self._loaded_extension().symmetric_static_stream_scheduled_adaln(
            x.contiguous(),
            adaln_scale,
            adaln_shift,
            int(modulation_span),
            float(epsilon),
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
        )

    def wam_stream_w4a4_adaln_linear_prevalidated(
        self,
        x: torch.Tensor,
        adaln_scale: torch.Tensor,
        adaln_shift: torch.Tensor,
        modulation_span: int,
        epsilon: float,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
    ) -> torch.Tensor:
        return self._loaded_w4a4_extension().symmetric_static_stream_scheduled_adaln(
            x.contiguous(),
            adaln_scale,
            adaln_shift,
            int(modulation_span),
            float(epsilon),
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
        )

    def wam_stream_gate_residual_prevalidated(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
        residual: torch.Tensor,
        gate: torch.Tensor,
        gate_span: int,
    ) -> torch.Tensor:
        """Fuse the DiT gate and residual update into a scheduled WAM GEMM."""
        return self._loaded_extension().symmetric_static_stream_scheduled_gate_residual(
            x.contiguous(),
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
            residual.contiguous(),
            gate,
            int(gate_span),
        )

    def wam_stream_w4a4_gate_residual_prevalidated(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
        residual: torch.Tensor,
        gate: torch.Tensor,
        gate_span: int,
    ) -> torch.Tensor:
        return self._loaded_w4a4_extension().symmetric_static_stream_scheduled_gate_residual(
            x.contiguous(),
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
            residual.contiguous(),
            gate,
            int(gate_span),
        )

    @staticmethod
    def pack_weight(qweight: torch.Tensor) -> torch.Tensor:
        return pack_signed_int4(qweight)

    def wam_linear(
        self,
        *,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        row_gains: torch.Tensor,
        activation_bits: int,
        keep_output_dtype: str,
    ) -> torch.Tensor:
        if activation_bits not in (4, 8):
            raise ValueError("The CUTLASS WAM fast path requires A4 or A8 activations.")
        if keep_output_dtype != "input":
            raise ValueError("The CUTLASS WAM backend currently keeps the input dtype.")
        if activation_bits == 4:
            rows = x.numel() // x.shape[-1]
            gains = (
                torch.broadcast_to(row_gains, (*x.shape[:-1], 1)).reshape(rows).to(dtype=x.dtype).contiguous()
            )
            return self._loaded_w4a4_extension().symmetric_static(
                x.contiguous(),
                qweight,
                scales.reshape(-1).contiguous(),
                bias,
                input_scale,
                activation_scale.float().reshape(1).contiguous(),
                gains,
                False,
            )
        return symmetric_static_linear(
            x,
            qweight,
            scales,
            activation_scale=activation_scale,
            row_gains=row_gains,
            bias=bias,
            input_scale=input_scale,
            input_scale_is_inverse=False,
        )

    def wam_stream_linear(
        self,
        *,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        activation_bits: int,
        keep_output_dtype: str,
    ) -> torch.Tensor:
        if activation_bits not in (4, 8):
            raise ValueError("The CUTLASS WAM fast path requires A4 or A8 activations.")
        if keep_output_dtype != "input":
            raise ValueError("The CUTLASS WAM backend currently keeps the input dtype.")
        if activation_bits == 4:
            return self._loaded_w4a4_extension().symmetric_static_stream_scheduled(
                x.contiguous(),
                qweight,
                scales.reshape(-1).contiguous(),
                bias,
                input_scale,
                activation_scale.float().reshape(1).contiguous(),
                stream_gains.float().reshape(1, -1).contiguous(),
                int(stream_span),
                0,
                False,
            )
        return symmetric_static_stream_linear(
            x,
            qweight,
            scales,
            activation_scale=activation_scale,
            stream_gains=stream_gains,
            stream_span=stream_span,
            bias=bias,
            input_scale=input_scale,
            input_scale_is_inverse=False,
        )

    def wam_stream_w4a4_preprocess_linear_prevalidated(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
        rotation_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse signed Hadamard, WAM A4 packing and the S4 Tensor-Core GEMM."""

        return self._loaded_w4a4_extension().symmetric_static_stream_scheduled_preprocess(
            x.contiguous(),
            rotation_signs,
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
        )

    def wam_stream_w4a4_preprocess_adaln_linear_prevalidated(
        self,
        x: torch.Tensor,
        adaln_scale: torch.Tensor,
        adaln_shift: torch.Tensor,
        modulation_span: int,
        epsilon: float,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
        rotation_signs: torch.Tensor,
    ) -> torch.Tensor:
        return self._loaded_w4a4_extension().symmetric_static_stream_scheduled_adaln_preprocess(
            x.contiguous(),
            adaln_scale,
            adaln_shift,
            int(modulation_span),
            float(epsilon),
            rotation_signs,
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
        )

    def wam_stream_w4a4_preprocess_gate_residual_prevalidated(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        input_scale_inverse: torch.Tensor,
        activation_scales: torch.Tensor,
        stream_gains: torch.Tensor,
        stream_span: int,
        timestep: int,
        residual: torch.Tensor,
        gate: torch.Tensor,
        gate_span: int,
        rotation_signs: torch.Tensor,
    ) -> torch.Tensor:
        return self._loaded_w4a4_extension().symmetric_static_stream_scheduled_gate_residual_preprocess(
            x.contiguous(),
            rotation_signs,
            qweight,
            scales,
            bias,
            input_scale_inverse,
            activation_scales,
            stream_gains,
            int(stream_span),
            int(timestep),
            True,
            residual.contiguous(),
            gate,
            int(gate_span),
        )
