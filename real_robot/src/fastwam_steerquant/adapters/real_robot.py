"""FastWAMJoint adapter for the released AgileX dual-arm W4A8 policies.

The model implementation comes from a pinned checkout of the public FastWAM
repository. Task-specific preprocessing, statistics and cached text conditions
are shipped with this subproject. No ROS, robot SDK or actuator code is imported.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch

from . import preprocessing
from .robot_core import DenseJointCoreMixin


FASTWAM_URL = "https://github.com/yuantianyuan01/FastWAM.git"
FASTWAM_REVISION = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    path = Path(path)
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise ValueError(f"SHA256 mismatch or missing expected hash: {path}")


def activate_fastwam(root, expected_revision=FASTWAM_REVISION):
    """Import FastWAM only from the clean, pinned public checkout."""
    root = Path(root).expanduser().resolve()
    source = (root / "src").resolve()
    package = source / "fastwam"
    if not (package / "models/wan22/fastwam_joint.py").is_file():
        raise FileNotFoundError(f"FastWAM checkout is incomplete: {root}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", "src/fastwam"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot verify FastWAM checkout at {root}") from exc
    if revision != expected_revision or dirty:
        raise RuntimeError(
            f"FastWAM must be clean at {expected_revision}; got revision={revision}, dirty={bool(dirty)}"
        )
    for name, module in tuple(sys.modules.items()):
        if name == "fastwam" or name.startswith("fastwam."):
            location = getattr(module, "__file__", None)
            locations = [location] if location else list(getattr(module, "__path__", []))
            if not locations or any(not Path(p).resolve().is_relative_to(source) for p in locations):
                raise RuntimeError(f"Another {name} is already imported; use a fresh process.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    module = importlib.import_module("fastwam")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Imported FastWAM does not come from the pinned checkout.")
    return revision


def public_fastwam_config(value):
    """Map only the audited qi namespace used in deployment metadata."""
    if isinstance(value, dict):
        return {key: public_fastwam_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_fastwam_config(item) for item in value]
    if isinstance(value, str) and value.startswith("qi."):
        allowed = (
            "qi.datasets.lerobot.processors.",
            "qi.datasets.lerobot.transforms.",
            "qi.datasets.lerobot.utils.",
            "qi.runtime.",
        )
        if not value.startswith(allowed):
            raise ValueError(f"Unsupported private qi target in deployment metadata: {value}")
        return "fastwam." + value[len("qi."):]
    return value


def build_structure(architecture, *, device="cpu"):
    """Construct the FastWAMJoint structure without loading pretrained weights."""
    if str(device) != "cpu":
        raise ValueError("Real robot builder requires construct_device='cpu', not meta/CUDA.")
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    class SteerQuantRealJoint(DenseJointCoreMixin, FastWAMJoint):
        pass

    arch = copy.deepcopy(architecture)
    if arch["video_dit_config"].get("action_conditioned", False):
        raise ValueError("Only unconditioned FastWAMJoint video expert is supported.")
    with torch.device("cpu"):
        video = WanVideoDiT(**arch["video_dit_config"]).to(dtype=torch.bfloat16)
        action = ActionDiT(**arch["action_dit_config"]).to(dtype=torch.bfloat16)
        mot = MoT({"video": video, "action": action},
                  mot_checkpoint_mixed_attn=arch["mot_checkpoint_mixed_attn"])
        vae = WanVideoVAE38().to(dtype=torch.bfloat16)
        scheduler_args = {}
        for expert in ("video", "action"):
            for key, value in arch[f"{expert}_scheduler"].items():
                scheduler_args[f"{expert}_{key}"] = value
        model = SteerQuantRealJoint(
            video_expert=video, action_expert=action, mot=mot, vae=vae,
            text_encoder=None, tokenizer=None,
            text_dim=arch["video_dit_config"]["text_dim"],
            proprio_dim=arch["proprio_dim"], device="cpu", torch_dtype=torch.bfloat16,
            loss_lambda_video=arch["loss"]["lambda_video"],
            loss_lambda_action=arch["loss"]["lambda_action"], **scheduler_args)
    return model.eval().requires_grad_(False)


def _sync_vae_scale_device(model):
    """Move public FastWAM's plain VAE scale tensors before Graph capture."""
    vae = model.vae
    if not isinstance(getattr(vae, "mean", None), torch.Tensor) or not isinstance(
            getattr(vae, "std", None), torch.Tensor):
        raise ValueError("FastWAM VAE must expose tensor mean/std scale values.")
    parameter = next(vae.model.parameters())
    vae.mean = vae.mean.to(device=parameter.device, dtype=parameter.dtype)
    vae.std = vae.std.to(device=parameter.device, dtype=parameter.dtype)
    vae.scale = [vae.mean, vae.std.reciprocal()]


class RealRobotAdapter:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.task = self.config["real_robot"]
        self.fastwam_revision = activate_fastwam(
            self.config["fastwam_root"], self.config.get("fastwam_revision", FASTWAM_REVISION)
        )
        for key in ("dataset_stats", "context_cache"):
            verify_file(self.config["assets"][key], self.task["asset_sha256"][key])
        if (self.config["num_calls"] != 10 or self.config.get("sigma_shift") is not None
                or self.config["model"] != {
                    "dtype": "bfloat16", "action_horizon": 32, "num_video_frames": 9
                }):
            raise ValueError("This adapter requires BF16, 10 steps, 32 actions, 9 video frames, shift=None.")
        from hydra.utils import instantiate
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        processor_config = public_fastwam_config(self.task["processor"])
        self.processor = instantiate(processor_config)
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(self.config["assets"]["dataset_stats"]))
        self.processor.eval()
        self.context, self.context_mask = preprocessing.load_cached_context(
            Path(self.config["assets"]["context_cache"]).parent,
            preprocessing.format_prompt(self.task["prompt"]), 128)
        if self.context.shape != (128, 4096) or not torch.isfinite(self.context).all():
            raise ValueError("Expected finite [128,4096] cached context.")

    def build_model(self, model_config, *, device="cpu"):
        self.validate_deployment_config(model_config)
        return build_structure(model_config["real_robot"]["architecture"], device=device)

    def validate_deployment_config(self, model_config):
        for key in ("num_calls", "sigma_shift", "model", "expected_sites", "stream_config"):
            if model_config[key] != self.config[key]:
                raise ValueError(f"Deployment and adapter disagree on {key}.")
        for key in ("asset_sha256", "architecture", "prompt", "task",
                    "source_checkpoint_sha256", "vae_checkpoint_sha256"):
            if model_config["real_robot"][key] != self.task[key]:
                raise ValueError(f"Deployment and adapter disagree on {key}.")
        if public_fastwam_config(model_config["real_robot"]["processor"]) != public_fastwam_config(
                self.task["processor"]):
            raise ValueError("Deployment and adapter disagree on processor.")

    def finalize_model(self, model, *, device):
        model.device = torch.device(device)
        model.torch_dtype = torch.bfloat16
        _sync_vae_scale_device(model)
        model.video_expert.freqs = tuple(x.to(device=device) for x in model.video_expert.freqs)
        model.action_expert.freqs = model.action_expert.freqs.to(device=device)

    def infer_kwargs(self, model, record, *, seed):
        import numpy as np
        if record.get("task") != self.task["task"] or record.get("prompt") != self.task["prompt"]:
            raise ValueError("Observation task/prompt differs from adapter.")
        channel_order = record.get("image_channel_order", "native")
        if channel_order != "native":
            raise ValueError(
                "Real-robot observations must preserve the native ROS/HDF channel order; "
                "do not apply an RGB/BGR channel swap."
            )
        if self.config.get("dataset", {}).get("channel_policy") != "as_stored":
            raise ValueError("Deployment metadata does not approve the native/as-stored channel contract.")
        images = {}
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            array = np.asarray(record["images"][name])
            if array.dtype != np.uint8 or array.shape != (480, 640, 3):
                raise ValueError(f"{name} must be uint8 [480,640,3] with the declared channel convention.")
            images[name] = array
        state = np.asarray(record["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("State must contain 14 finite physical-coordinate values.")
        obs = preprocessing.WAMObservation(**images, state=state, prompt=record["prompt"])
        image = preprocessing.preprocess_real_images(obs, model.device, model.torch_dtype)
        proprio = preprocessing.normalize_proprio(self.processor, state).unsqueeze(0)
        return dict(prompt=None, input_image=image, proprio=proprio,
                    context=self.context, context_mask=self.context_mask,
                    action_horizon=32, num_video_frames=9, num_inference_steps=10,
                    sigma_shift=None, text_cfg_scale=1.0, seed=seed, rand_device="cpu",
                    tiled=False)

    def denormalize_actions(self, action):
        return preprocessing.denormalize_actions(self.processor, action.detach().cpu())


def make_adapter(config):
    return RealRobotAdapter(config)
