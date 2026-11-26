"""LIBERO calibration input construction and matched Cosmos sampling.

Uses matched preprocessing and a five-call 2AB schedule.
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import torch
from cosmos_policy._src.imaginaire.functional.multi_step import order2_fn
from cosmos_policy._src.imaginaire.modules.res_sampler import get_rev_ts
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE,
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    get_t5_embedding_from_cache,
    prepare_images_for_model,
    rescale_proprio,
)
from cosmos_policy.utils.utils import duplicate_array


def load_calibration(path: Path, limit: int | None, *, suite: str = "libero") -> list[dict[str, Any]]:
    chunk_size = {"libero": 16, "robocasa": 32}[suite]
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    values = payload if isinstance(payload, (list, tuple)) else [payload]
    records: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise TypeError(f"{path}[{index}] is not a dictionary")
        required = {"observation", "task_label", "action_chunk"}
        missing = required.difference(value)
        if missing:
            raise KeyError(f"{path}[{index}] is missing keys: {sorted(missing)}")
        observation = value["observation"]
        obs_missing = {"primary_image", "wrist_image", "proprio"}.difference(observation)
        if suite == "robocasa" and "secondary_image" not in observation:
            obs_missing.add("secondary_image")
        if obs_missing:
            raise KeyError(f"{path}[{index}].observation is missing keys: {sorted(obs_missing)}")
        actions = np.asarray(value["action_chunk"], dtype=np.float32)
        if actions.shape != (chunk_size, ACTION_DIM):
            raise ValueError(
                f"{path}[{index}] action_chunk has shape {actions.shape}; expected ({chunk_size}, {ACTION_DIM})"
            )
        metadata = dict(value.get("metadata", {}))
        sample_index = int(metadata.get("sample_index", index))
        records.append(
            {
                "observation_id": f"calibration:{sample_index:03d}",
                "observation": observation,
                "task_label": str(value["task_label"]),
                "action_chunk": actions,
                "metadata": metadata,
            }
        )
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"No calibration records loaded from {path}")
    return records


def build_eval_config(args: argparse.Namespace, checkpoint: Path) -> SimpleNamespace:
    suite = getattr(args, "suite", "libero")
    return SimpleNamespace(
        suite=suite,
        config=args.config,
        ckpt_path=str(checkpoint),
        config_file=args.config_file,
        use_third_person_image=True,
        num_third_person_images=2 if suite == "robocasa" else 1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        # Training HDF5 already stores quality-95 JPEG bytes. The calibration
        # records contain their decoded pixels, so recompressing would add a
        # second, unintended JPEG pass.
        use_jpeg_compression=bool(args.recompress_jpeg),
        trained_with_image_aug=True,
        use_variance_scale=False,
        chunk_size=args.chunk_size,
        unnormalize_actions=True,
        quantization_enabled=False,
    )


def build_policy_data_batch(
    cfg: SimpleNamespace,
    model: torch.nn.Module,
    dataset_stats: dict[str, np.ndarray],
    observation: dict[str, Any],
    task_label: str,
) -> tuple[dict[str, Any], int]:
    device = next(model.parameters()).device
    if cfg.suite not in ("libero", "robocasa"):
        raise ValueError(f"Unsupported calibration suite: {cfg.suite}")
    robocasa = cfg.suite == "robocasa"
    images = [observation["wrist_image"], observation["primary_image"]]
    if robocasa:
        images.append(observation["secondary_image"])
    processed = prepare_images_for_model(images, cfg)
    wrist_image, primary_image = processed[:2]
    proprio = np.asarray(observation["proprio"], dtype=np.float32)
    if cfg.normalize_proprio:
        proprio = rescale_proprio(proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0)

    blank_image = np.zeros_like(primary_image)
    blank_duplicated = duplicate_array(blank_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    wrist_duplicated = duplicate_array(wrist_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    primary_duplicated = duplicate_array(primary_image, total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
    image_sequence = [
        np.expand_dims(blank_image, axis=0),
        blank_duplicated.copy(),  # current proprio: slot 1
        wrist_duplicated,  # current wrist: slot 2
        primary_duplicated,  # current primary: slot 3
        blank_duplicated.copy(),  # action: slot 4
        blank_duplicated.copy(),  # future proprio: slot 5
        wrist_duplicated.copy(),  # future wrist: slot 6
        primary_duplicated.copy(),  # future primary: slot 7
        blank_duplicated.copy(),  # value: slot 8
    ]
    if robocasa:
        secondary = duplicate_array(processed[2], total_num_copies=COSMOS_TEMPORAL_COMPRESSION_FACTOR)
        image_sequence.insert(4, secondary)
        image_sequence.insert(9, secondary.copy())
    raw_image_sequence = np.concatenate(image_sequence, axis=0)[None]
    raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))
    video = torch.from_numpy(np.ascontiguousarray(raw_image_sequence)).to(device=device, dtype=torch.uint8)
    proprio_tensor = (
        torch.from_numpy(np.ascontiguousarray(proprio)).reshape(1, -1).to(device=device, dtype=torch.bfloat16)
    )
    text_embedding = get_t5_embedding_from_cache(task_label)
    if not isinstance(text_embedding, torch.Tensor):
        text_embedding = torch.as_tensor(text_embedding)
    text_embedding = text_embedding.to(device=device, dtype=torch.bfloat16)

    def index_tensor(value: int) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.int64, device=device)

    action_latent_idx = 5 if robocasa else 4
    data_batch = {
        "dataset_name": "video_data",
        "video": video,
        "t5_text_embeddings": text_embedding.repeat(1, 1, 1),
        "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(
            (1, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16, device=device
        ),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio_tensor,
        "current_proprio_latent_idx": index_tensor(1),
        "current_wrist_image_latent_idx": index_tensor(2),
        "current_wrist_image2_latent_idx": index_tensor(-1),
        "current_image_latent_idx": index_tensor(3),
        "current_image2_latent_idx": index_tensor(4 if robocasa else -1),
        "action_latent_idx": index_tensor(action_latent_idx),
        "future_proprio_latent_idx": index_tensor(6 if robocasa else 5),
        "future_wrist_image_latent_idx": index_tensor(7 if robocasa else 6),
        "future_wrist_image2_latent_idx": index_tensor(-1),
        "future_image_latent_idx": index_tensor(8 if robocasa else 7),
        "future_image2_latent_idx": index_tensor(9 if robocasa else -1),
        "value_latent_idx": index_tensor(10 if robocasa else 8),
    }
    return data_batch, action_latent_idx


# Preserve the public LIBERO helper used by earlier experiment scripts.
build_libero_data_batch = build_policy_data_batch


def differentiable_sample(
    model: torch.nn.Module,
    data_batch: dict[str, Any],
    collector: Any,
    *,
    seed: int,
    num_steps: int,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("This experiment expects at least two denoising steps")
    model._normalize_video_databatch_inplace(data_batch)
    model._augment_image_dim_inplace(data_batch)
    input_key = model.input_image_key if model.is_image_batch(data_batch) else model.input_data_key
    _, raw_t, raw_h, raw_w = data_batch[input_key].shape[-4:]
    state_shape = (
        model.config.state_ch,
        model.tokenizer.get_latent_num_frames(raw_t),
        raw_h // model.tokenizer.spatial_compression_factor,
        raw_w // model.tokenizer.spatial_compression_factor,
    )
    x0_fn = model.get_x0_fn_from_batch(data_batch, guidance=1.5, is_negative_prompt=False)
    in_dtype = torch.float32
    sigma_max = float(model.sde.sigma_max)
    sigma_min = float(model.sde.sigma_min)
    noisy = misc.arch_invariant_rand((1,) + state_shape, in_dtype, next(model.parameters()).device, seed)
    noisy = (noisy * sigma_max).requires_grad_(True)

    # CosmosPolicySampler subtracts one NFE when sample_clean=True, then performs
    # one final clean denoiser call. Thus a user request of 5 yields 4 solver
    # calls plus the final clean call.
    solver_nfe = num_steps - 1
    sigmas = get_rev_ts(sigma_min, sigma_max, solver_nfe, 7.0).to(noisy.device)
    ones = torch.ones(noisy.shape[0], device=noisy.device, dtype=torch.float64)
    state = noisy
    x0_history: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def tracked_x0(current: torch.Tensor, sigma: torch.Tensor, call_index: int) -> torch.Tensor:
        collector.begin_denoise_call(call_index, sigma)
        if getattr(collector, "checkpoint_denoisers", False):
            def evaluate(x, s):
                return x0_fn(x.to(in_dtype), s.to(in_dtype)).to(torch.float64)

            return collector.checkpoint_call(evaluate, current, sigma, call_index)
        return x0_fn(current.to(in_dtype), sigma.to(in_dtype)).to(torch.float64)

    for call_index in range(solver_nfe):
        sigma_current = sigmas[call_index] * ones
        sigma_next = sigmas[call_index + 1] * ones
        x0_pred = tracked_x0(state, sigma_current, call_index)
        state, x0_history = order2_fn(state, sigma_current, sigma_next, x0_pred, x0_history)

    final = tracked_x0(state, sigmas[-1] * ones, solver_nfe)
    if collector._network_calls_seen != num_steps:
        raise RuntimeError(
            f"Expected {num_steps} generator calls, observed {collector._network_calls_seen}; "
            "the model may be using CFG or a different scheduler."
        )
    return final.to(in_dtype)


def unnormalize_actions_torch(actions: torch.Tensor, dataset_stats: dict[str, np.ndarray]) -> torch.Tensor:
    action_min = torch.as_tensor(dataset_stats["actions_min"], device=actions.device, dtype=actions.dtype)
    action_max = torch.as_tensor(dataset_stats["actions_max"], device=actions.device, dtype=actions.dtype)
    return 0.5 * (actions + 1.0) * (action_max - action_min) + action_min


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one .pt checkpoint under {path}; found {len(candidates)}")
    return candidates[0]


def generate_matched(model, data_batch, *, seed, num_steps):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        generated, _ = model.generate_samples_from_batch(
            data_batch,
            n_sample=1,
            num_steps=num_steps,
            seed=seed,
            is_negative_prompt=False,
            use_variance_scale=False,
            return_orig_clean_latent_frames=True,
        )
    return generated
