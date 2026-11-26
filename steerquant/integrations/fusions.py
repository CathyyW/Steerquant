# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from NVIDIA Cosmos Policy; adapted for SteerQuant.
"""Instance-local DiT fusion adapter; the upstream package stays unchanged."""

from __future__ import annotations
from types import MethodType
from typing import Optional
from cosmos_policy._src.predict2.utils.kv_cache import KVCacheConfig
import torch
from torch import nn
import torch.amp as amp
from einops import rearrange
from cosmos_policy._src.predict2.networks.minimal_v4_dit import (
    Attention,
    Block,
    GPT2FeedForward,
    VideoSize,
    apply_rotary_pos_emb,
    NattenA2AAttnOp,
    NeighborhoodAttention,
    AttentionOpWithKVCache,
)


def GPT2FeedForward_forward_adaln_gate_residual(
    self,
    x: torch.Tensor,
    norm_layer: nn.Module,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Use W4A8 block fusions when provided by the quantized Linears."""
    broadcast_shape = (*adaln_scale.shape[:-1], *[1] * (x.ndim - adaln_scale.ndim), x.shape[-1])
    if getattr(self.layer1, "supports_fused_adaln", False):
        hidden = self.layer1.forward_adaln(x, adaln_scale, adaln_shift, norm_layer.eps)
    else:
        scale = adaln_scale.reshape(broadcast_shape).type_as(x)
        shift = adaln_shift.reshape(broadcast_shape).type_as(x)
        hidden = self.layer1(norm_layer(x) * (1 + scale) + shift)
    hidden = self.activation(hidden)
    if getattr(self.layer2, "supports_fused_gate_residual", False):
        return self.layer2.forward_gate_residual(hidden, x, gate)
    broadcast_shape = (*gate.shape[:-1], *[1] * (x.ndim - gate.ndim), x.shape[-1])
    return x + gate.reshape(broadcast_shape).type_as(x) * self.layer2(hidden)


def Attention__apply_norm_and_rotary_pos_emb(self, q, k, v, rope_emb, *, already_normalized: bool = False):
    if not already_normalized:
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
    original_dtype = q.dtype
    if self.is_selfattn and rope_emb is not None:
        if self.use_wan_fp32_strategy:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
        q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
        k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
        if self.use_wan_fp32_strategy:
            q = q.to(original_dtype)
            k = k.to(original_dtype)
    return (q, k, v)


def Attention_compute_qkv_adaln(
    self,
    x_B_T_H_W_D: torch.Tensor,
    norm_layer: nn.Module,
    adaln_scale_B_T_D: torch.Tensor,
    adaln_shift_B_T_D: torch.Tensor,
    context: torch.Tensor,
    rope_emb=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse cross-attention query AdaLN into its W4A8 producer."""
    B, T, H, W, _ = x_B_T_H_W_D.shape
    if getattr(self.q_proj, "supports_fused_adaln", False):
        q = self.q_proj.forward_adaln(
            x_B_T_H_W_D,
            adaln_scale_B_T_D.type_as(x_B_T_H_W_D),
            adaln_shift_B_T_D.type_as(x_B_T_H_W_D),
            norm_layer.eps,
        )
        q = rearrange(q, "b t h w d -> b (t h w) d")
    else:
        scale = rearrange(adaln_scale_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        shift = rearrange(adaln_shift_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        normalized = norm_layer(x_B_T_H_W_D) * (1 + scale) + shift
        q = self.q_proj(rearrange(normalized, "b t h w d -> b (t h w) d"))
    k = self.k_proj(context)
    v = self.v_proj(context)
    q, k, v = map(
        lambda tensor: rearrange(tensor, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
        (q, k, v),
    )
    return self._apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)


def Attention_compute_attention(
    self,
    q,
    k,
    v,
    video_size: Optional[VideoSize] = None,
    kv_cache_cfg: Optional[KVCacheConfig] = None,
    residual: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
):
    additional_args = {}
    if isinstance(self.attn_op, (NattenA2AAttnOp, NeighborhoodAttention)) or self.backend == "i4":
        additional_args["video_size"] = video_size
    if isinstance(self.attn_op, AttentionOpWithKVCache):
        additional_args["kv_cache_cfg"] = kv_cache_cfg
    result = self.attn_op(q, k, v, **additional_args)
    if (
        residual is not None
        and gate is not None
        and isinstance(self.output_dropout, nn.Identity)
        and getattr(self.output_proj, "supports_fused_gate_residual", False)
    ):
        return self.output_proj.forward_gate_residual(result, residual, gate)
    result = self.output_dropout(self.output_proj(result))
    if residual is not None and gate is not None:
        batch, sequence, width = residual.shape
        gate_rows = gate.numel() // width
        gate_span = sequence // (gate_rows // batch)
        expanded_gate = gate.reshape(batch, gate_rows // batch, 1, width).expand(
            batch, gate_rows // batch, gate_span, width
        )
        return residual + expanded_gate.reshape_as(residual).type_as(result) * result
    return result


def Attention_forward_gate_residual(
    self,
    x: torch.Tensor,
    context: Optional[torch.Tensor],
    residual: torch.Tensor,
    gate: torch.Tensor,
    rope_emb: Optional[torch.Tensor] = None,
    video_size: Optional[VideoSize] = None,
    kv_cache_cfg: Optional[KVCacheConfig] = None,
) -> torch.Tensor:
    q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
    return self.compute_attention(
        q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg, residual=residual, gate=gate
    )


def Attention_forward_adaln_gate_residual(
    self,
    x_B_T_H_W_D: torch.Tensor,
    norm_layer: nn.Module,
    adaln_scale_B_T_D: torch.Tensor,
    adaln_shift_B_T_D: torch.Tensor,
    context: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    rope_emb: Optional[torch.Tensor] = None,
    video_size: Optional[VideoSize] = None,
    kv_cache_cfg: Optional[KVCacheConfig] = None,
) -> torch.Tensor:
    q, k, v = self.compute_qkv_adaln(
        x_B_T_H_W_D, norm_layer, adaln_scale_B_T_D, adaln_shift_B_T_D, context, rope_emb=rope_emb
    )
    return self.compute_attention(
        q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg, residual=residual, gate=gate
    )


def Block_forward(
    self,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
    adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    kv_cache_cfg: Optional[KVCacheConfig] = None,
) -> torch.Tensor:
    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
    with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(
                3, dim=-1
            )
    shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
    B, T, H, W, D = x_B_T_H_W_D.shape

    def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
        return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

    normalized_x_B_T_H_W_D = _fn(
        x_B_T_H_W_D, self.layer_norm_self_attn, scale_self_attn_B_T_1_1_D, shift_self_attn_B_T_1_1_D
    )
    video_size = VideoSize(T=T, H=H, W=W)
    if self.cp_size is not None and self.cp_size > 1:
        video_size = VideoSize(T=T * self.cp_size, H=H, W=W)
    x_B_T_H_W_D = rearrange(
        self.self_attn.forward_gate_residual(
            rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
            None,
            rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
            gate_self_attn_B_T_D.type_as(x_B_T_H_W_D),
            rope_emb=rope_emb_L_1_1_D,
            video_size=video_size,
            kv_cache_cfg=kv_cache_cfg,
        ),
        "b (t h w) d -> b t h w d",
        t=T,
        h=H,
        w=W,
    )
    if isinstance(crossattn_emb, torch.Tensor):
        x_B_T_H_W_D = rearrange(
            self.cross_attn.forward_adaln_gate_residual(
                x_B_T_H_W_D,
                self.layer_norm_cross_attn,
                scale_cross_attn_B_T_D,
                shift_cross_attn_B_T_D,
                crossattn_emb,
                rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                gate_cross_attn_B_T_D.type_as(x_B_T_H_W_D),
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
    else:
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D, self.layer_norm_cross_attn, scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D
        )
        result_B_T_H_W_D = rearrange(
            self.cross_attn(
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D
    return self.mlp.forward_adaln_gate_residual(
        x_B_T_H_W_D,
        self.layer_norm_mlp,
        scale_mlp_B_T_D,
        shift_mlp_B_T_D,
        gate_mlp_B_T_D.type_as(x_B_T_H_W_D),
    )


def install_fusions(net):
    for module in net.modules():
        if type(module) is GPT2FeedForward:
            module.forward_adaln_gate_residual = MethodType(
                GPT2FeedForward_forward_adaln_gate_residual, module
            )
        if type(module) is Attention:
            module._apply_norm_and_rotary_pos_emb = MethodType(
                Attention__apply_norm_and_rotary_pos_emb, module
            )
        if type(module) is Attention:
            module.compute_qkv_adaln = MethodType(Attention_compute_qkv_adaln, module)
        if type(module) is Attention:
            module.compute_attention = MethodType(Attention_compute_attention, module)
        if type(module) is Attention:
            module.forward_gate_residual = MethodType(Attention_forward_gate_residual, module)
        if type(module) is Attention:
            module.forward_adaln_gate_residual = MethodType(Attention_forward_adaln_gate_residual, module)
        if type(module) is Block:
            module.forward = MethodType(Block_forward, module)
