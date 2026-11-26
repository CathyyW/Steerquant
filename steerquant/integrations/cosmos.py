"""Local-asset Cosmos-Policy adapter for LIBERO and RoboCasa."""

import importlib
import importlib.util
import os
import sys


def configure_platform(suite):
    """Initialize upstream's argv-based constants before any model imports.

    Paths can contain another benchmark's name. Only the explicit platform
    should select constants; a previously imported different platform is an error.
    """
    if suite not in ("libero", "robocasa"):
        raise ValueError(f"Unknown platform: {suite}")
    original = sys.argv
    try:
        sys.argv = ["steerquant", "--suite", suite]
        constants = importlib.import_module("cosmos_policy.constants")
    finally:
        sys.argv = original
    expected_chunk = 32 if suite == "robocasa" else 16
    if constants.ROBOT_PLATFORM.lower() != suite or constants.NUM_ACTIONS_CHUNK != expected_chunk:
        raise RuntimeError("Cosmos constants already selected another platform; launch a fresh process")


def init_t5_text_embeddings_cache(*args, **kwargs):
    from cosmos_policy.experiments.robot.cosmos_utils import init_t5_text_embeddings_cache as initialize
    return initialize(*args, **kwargs)


def load_dataset_stats(*args, **kwargs):
    from cosmos_policy.experiments.robot.cosmos_utils import load_dataset_stats as load
    return load(*args, **kwargs)


def get_model(cfg):
    from cosmos_policy._src.imaginaire.utils import checkpoint_db
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    # Upstream registers all experiments at import time, resolving unrelated
    # pretrained checkpoints. This adapter supplies the selected local policy
    # during registration; the loader subsequently selects cfg.config.
    original = checkpoint_db.get_checkpoint_path
    redundant = {
        "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
        "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt",
    }

    def local_checkpoint(uri, *args, **kwargs):
        return str(cfg.ckpt_path) if uri in redundant else original(uri, *args, **kwargs)

    checkpoint_db.get_checkpoint_path = local_checkpoint
    try:
        model, config = load_model_from_checkpoint(
            experiment_name=cfg.config,
            s3_checkpoint_dir=str(cfg.ckpt_path),
            config_file=cfg.config_file,
            load_ema_to_reg=False,
            experiment_opts=[f"model.config.tokenizer.vae_pth={os.environ['COSMOS_POLICY_TOKENIZER']}"],
        )
    finally:
        checkpoint_db.get_checkpoint_path = original
    model.eval().to("cuda")
    return model, config


def prepare_paths(args):
    """Validate assets before importing/constructing the model."""
    model_dir = args.model_dir.expanduser().resolve()
    suite = getattr(args, "suite", "libero")
    if suite not in ("libero", "robocasa"):
        raise ValueError("Platform must be libero or robocasa; pass LIBERO task suites separately")
    model_label = {"libero": "LIBERO", "robocasa": "RoboCasa"}[suite]
    args.checkpoint = model_dir / f"Cosmos-Policy-{model_label}-Predict2-2B.pt"
    args.dataset_stats = model_dir / f"{suite}_dataset_statistics.json"
    args.t5_embeddings = model_dir / f"{suite}_t5_embeddings.pkl"
    args.tokenizer_checkpoint = args.tokenizer.expanduser().resolve()
    for path in (args.checkpoint, args.dataset_stats, args.t5_embeddings, args.tokenizer_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model asset: {path}. See README model downloads.")
    os.environ["COSMOS_POLICY_CKPT"] = str(args.checkpoint)
    os.environ["COSMOS_POLICY_TOKENIZER"] = str(args.tokenizer_checkpoint)
    os.environ["DETERMINISTIC"] = "True"
    args.config = (
        "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"
        if suite == "robocasa" else "cosmos_predict2_2b_480p_libero__inference_only"
    )
    # Resolve the installed package, not the user's current working directory.
    spec = importlib.util.find_spec("cosmos_policy")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Install the pinned cosmos-policy dependency first")
    from pathlib import Path
    args.config_file = str(Path(next(iter(spec.submodule_search_locations))) / "config/config.py")
    configure_platform(suite)
    return args
