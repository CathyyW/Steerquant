"""Run the upstream LIBERO evaluation with a SteerQuant checkpoint."""

import argparse
import os
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="SteerQuant calibration checkpoint")
    p.add_argument(
        "--suite",
        dest="task_suite",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        default="libero_spatial",
    )
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--backend", choices=("cutlass", "reference"), default="cutlass")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    p.add_argument("--cuda-graph", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    args.suite = "libero"
    checkpoint_path = args.checkpoint.expanduser().resolve()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from .integrations.cosmos import prepare_paths, get_model

    prepare_paths(args)
    from .integrations.libero import prepare_libero

    prepare_libero()
    from cosmos_policy.experiments.robot.libero import run_libero_eval as evaluation
    from .wam_pipeline import WAMCalibrationCheckpoint
    from .runtime import quantize_model_for_inference

    ckpt = WAMCalibrationCheckpoint.load(checkpoint_path)
    cfg = evaluation.PolicyEvalConfig(
        config=args.config,
        config_file=args.config_file,
        ckpt_path=str(args.checkpoint),
        dataset_stats_path=str(args.dataset_stats),
        t5_text_embeddings_path=str(args.t5_embeddings),
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        trained_with_image_aug=True,
        chunk_size=16,
        num_open_loop_steps=16,
        task_suite_name=args.task_suite,
        num_trials_per_task=args.trials,
        seed=args.seed,
        randomize_seed=False,
        deterministic=True,
        data_collection=False,
        use_variance_scale=False,
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        local_log_dir=str(args.output_dir),
        run_id_note=f"steerquant_w4a{ckpt.activation_bits}_{args.backend}",
    )
    cfg.quantization_enabled = True
    cfg.quantization_algo = "wam"
    cfg.quantization_weight_bits = ckpt.weight_bits
    cfg.quantization_activation_bits = ckpt.activation_bits
    cfg.quantization_backend = args.backend
    cfg.quantization_calibration_path = str(checkpoint_path)
    cfg.dit_cuda_graph = args.cuda_graph

    def load_quantized(config):
        model, model_config = get_model(config)
        if args.backend == "cutlass":
            from .integrations.fusions import install_fusions

            install_fusions(model.net)
        return quantize_model_for_inference(model, cfg), model_config

    original_get_model = evaluation.get_model
    evaluation.get_model = load_quantized
    # Close each task's renderer while EGL is still alive, including on errors.
    original_env_factory = evaluation.get_libero_env
    active_environments = []

    def create_environment(*positional, **keywords):
        while active_environments:
            active_environments.pop().close()
        env, description = original_env_factory(*positional, **keywords)
        active_environments.append(env)
        return env, description

    evaluation.get_libero_env = create_environment
    # LIBERO's published initial-state files contain NumPy arrays, while
    # torch >= 2.6 defaults to weights-only loading.
    import numpy as np
    import torch

    numpy_types = [
        (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]
    try:
        with torch.serialization.safe_globals(numpy_types):
            evaluation.eval_libero(cfg)
    finally:
        while active_environments:
            active_environments.pop().close()
        evaluation.get_libero_env = original_env_factory
        evaluation.get_model = original_get_model


if __name__ == "__main__":
    main()
