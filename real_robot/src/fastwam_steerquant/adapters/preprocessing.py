"""AgileX Cobot Magic observation preprocessing used by the released policy."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

@dataclass
class WAMObservation:
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray
    prompt: str


def dtype_from_mixed_precision(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision: {mixed_precision}")


def format_prompt(task_prompt: str) -> str:
    return DEFAULT_PROMPT.format(task=task_prompt)


def load_cached_context(cache_dir: str | Path, formatted_prompt: str, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir)
    cache_hash = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_hash}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Run scripts/precompute_text_embeds.py for this prompt, or set use_text_encoder=true."
        )
    payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    context = payload["context"]
    context_mask = payload["mask"].bool()
    if context.ndim != 2 or context_mask.ndim != 1:
        raise ValueError(f"Invalid cached context shapes in {cache_path}: {context.shape}, {context_mask.shape}")
    if context.shape[0] != context_len or context_mask.shape[0] != context_len:
        raise ValueError(
            f"Cached context length mismatch in {cache_path}: "
            f"context={context.shape[0]}, mask={context_mask.shape[0]}, expected={context_len}"
        )
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask, dtype=torch.bool)
    return context, context_mask


def native_hwc_to_float_chw(native_uint8: np.ndarray) -> torch.Tensor:
    """Preserve the experiment's native three-channel byte order."""
    if native_uint8.ndim != 3 or native_uint8.shape[2] != 3:
        raise ValueError(f"Expected native uint8 image as [H,W,3], got {native_uint8.shape}")
    native_uint8 = np.ascontiguousarray(native_uint8).copy()
    return torch.from_numpy(native_uint8).permute(2, 0, 1).to(torch.float32) / 255.0


def preprocess_real_images(observation: WAMObservation, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cam_high = transforms_F.resize(
        native_hwc_to_float_chw(observation.cam_high),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        native_hwc_to_float_chw(observation.cam_left_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        native_hwc_to_float_chw(observation.cam_right_wrist),
        [240, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_high = transforms_F.resize(
        cam_high,
        [256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = transforms_F.resize(
        cam_left,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = transforms_F.resize(
        cam_right,
        [128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat([cam_left, cam_right], dim=-1)
    image = torch.cat([cam_high, bottom], dim=-2)
    if image.shape != (3, 384, 320):
        raise RuntimeError(f"Unexpected three-camera mosaic shape: {tuple(image.shape)}")
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(0).to(device=device, dtype=dtype)


def normalize_proprio(processor: Any, state: np.ndarray) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    expected_dim = int(processor.proprio_output_dim)
    if state_tensor.numel() != expected_dim:
        raise ValueError(f"Expected proprio/state dim {expected_dim}, got {state_tensor.numel()}")
    batch = {"state": {"default": state_tensor.view(1, -1)}}
    batch = processor.normalizer.forward(batch)
    return batch["state"]["default"].squeeze(0)


def denormalize_actions(processor: Any, action_norm: torch.Tensor) -> torch.Tensor:
    if action_norm.ndim == 3 and action_norm.shape[0] == 1:
        action_norm = action_norm.squeeze(0)
    if action_norm.ndim == 1:
        action_norm = action_norm.view(1, -1)
    if action_norm.ndim != 2:
        raise ValueError(f"Expected normalized action [T,D], got {tuple(action_norm.shape)}")
    dummy_state = torch.zeros(
        1,
        action_norm.shape[0],
        int(processor.proprio_output_dim),
        dtype=torch.float32,
        device=action_norm.device,
    )
    batch = {
        "action": {"default": action_norm.unsqueeze(0).to(torch.float32)},
        "state": {"default": dummy_state},
    }
    batch = processor.normalizer.backward(batch)
    return batch["action"]["default"].squeeze(0)
