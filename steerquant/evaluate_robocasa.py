"""RoboCasa FP / packed SteerQuant evaluation with mandatory backend smoke gating.

Uses Cosmos-Policy environment and rollout functions with SteerQuant quantization.
"""

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from .integrations.robocasa_protocol import (
    SCENES,
    TASKS,
    canonical,
    digest,
    make_manifest,
    normalize_metadata,
    summarize,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("outputs/robocasa_720")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def shared_json(path, value):
    """Create an immutable shared record or fail on a mismatch, under flock."""
    # JSON persists tuple-valued camera metadata as lists. Compare the same
    # serialization representation on both sides, preserving order and values.
    value = json.loads(canonical(value))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            previous = json.loads(path.read_text())
            if previous != value:
                difference = [k for k in set(previous) | set(value) if previous.get(k) != value.get(k)]
                write_json(path.with_suffix(f".mismatch.{os.getpid()}.json"), value)
                raise RuntimeError(f"Shared protocol/initial state mismatch: {path}; fields={difference}")
        else:
            write_json(path, value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fp", "w4a8", "w4a4"), required=True)
    parser.add_argument("--mode", choices=("smoke", "run", "check"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", dest="quant_checkpoint", type=Path, help="Required for W4A8/W4A4; unused for FP"
    )
    parser.add_argument("--output-dir", "--output-root", dest="output_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--check-only", action="store_true", help="Validate assets and platform without loading a model"
    )
    parser.set_defaults(suite="robocasa")
    return parser


def validate_args(args):
    if args.method == "fp" and args.quant_checkpoint is not None:
        raise ValueError("FP evaluation does not accept a quantized checkpoint")
    if args.method != "fp":
        if args.quant_checkpoint is None:
            raise ValueError("--checkpoint is required for W4A8/W4A4")
        args.quant_checkpoint = args.quant_checkpoint.expanduser().resolve()
        if not args.quant_checkpoint.is_file():
            raise FileNotFoundError(args.quant_checkpoint)
    args.output_root = args.output_root.expanduser().resolve()
    return args


def provenance(args, manifest, evaluation):
    import platform
    import torch
    import robocasa
    import robosuite
    import mujoco

    upstream = Path(evaluation.__file__).resolve().parents[3]
    paths = [
        Path(evaluation.__file__),
        Path(evaluation.__file__).with_name("robocasa_utils.py"),
        upstream / "experiments/robot/cosmos_utils.py",
        upstream / "_src/predict2/networks/minimal_v4_dit.py",
        Path(args.config_file),
    ]
    sources = {"cosmos_policy/" + str(p.relative_to(upstream)): file_hash(p) for p in paths}
    sources.update(
        {
            "steerquant/" + str(p.relative_to(PACKAGE_ROOT)): file_hash(p)
            for p in sorted(PACKAGE_ROOT.rglob("*"))
            if p.suffix in (".py", ".cu", ".cpp", ".h", ".cuh", ".json")
        }
    )
    inputs = dict(
        model=args.checkpoint,
        statistics=args.dataset_stats,
        text_embeddings=args.t5_embeddings,
        tokenizer=args.tokenizer_checkpoint,
        controller=Path(evaluation.__file__).with_name("robocasa_controller_configs.pkl"),
    )
    if args.quant_checkpoint:
        inputs["quantization"] = args.quant_checkpoint
    return dict(
        method=args.method,
        manifest_sha256=digest(manifest),
        source_sha256=digest(sources),
        sources=sources,
        input_sha256={name: file_hash(p) for name, p in inputs.items()},
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        robocasa=getattr(robocasa, "__version__", "unknown"),
        robosuite=robosuite.__version__,
        mujoco=mujoco.__version__,
    )


def make_config(args, evaluation):
    cfg = evaluation.PolicyEvalConfig(
        config=args.config,
        config_file=args.config_file,
        ckpt_path=str(args.checkpoint),
        dataset_stats_path=str(args.dataset_stats),
        t5_text_embeddings_path=str(args.t5_embeddings),
        layout_and_style_ids=repr(SCENES),
        num_trials_per_task=10,
        deterministic=True,
        deterministic_reset=True,
        data_collection=False,
        use_wandb=False,
        use_third_person_image=True,
        num_third_person_images=2,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        chunk_size=32,
        num_open_loop_steps=16,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_variance_scale=False,
        randomize_seed=False,
        seed=195,
    )
    cfg.dit_cuda_graph = False  # Install native-kernel guards before graph capture.
    cfg.dit_cuda_graph_warmup = 3
    cfg.quantization_enabled = args.method != "fp"
    if cfg.quantization_enabled:
        cfg.quantization_algo = "wam"
        cfg.quantization_backend = "cutlass"
        cfg.quantization_weight_bits = 4
        cfg.quantization_activation_bits = int(args.method[-1])
        cfg.quantization_calibration_path = str(args.quant_checkpoint)
    return cfg


def load_model(cfg):
    from .integrations.cosmos import get_model

    model, model_config = get_model(cfg)
    if cfg.quantization_enabled:
        from .integrations.fusions import install_fusions
        from .runtime import quantize_model_for_inference

        install_fusions(model.net)
        quantize_model_for_inference(model, cfg)
    return model, model_config


def install_kernel_guards(model, method):
    import torch
    from . import wam_runtime

    modules = [(n, m) for n, m in model.net.named_modules() if isinstance(m, wam_runtime.WAMFakeQuantLinear)]
    if method == "fp":
        if modules:
            raise RuntimeError("FP model unexpectedly has quantized WAM modules")
        return dict(module_count=0, backend="bf16"), Counter()
    if len(modules) != 280:
        raise RuntimeError(f"Expected 280 quantized Linear maps, got {len(modules)}")
    calls = Counter()

    def forbidden(*args, **kwargs):
        raise RuntimeError("Non-fused/reference WAM path forbidden in this experiment")

    wam_runtime.fake_quant_activation = forbidden
    backend = modules[0][1].backend
    backend.wam_linear = forbidden
    backend.wam_stream_linear = forbidden
    for name, module in modules:
        if module.backend.name != "cutlass" or module.qweight.dtype != torch.uint8:
            raise RuntimeError(f"Not a packed CUTLASS module: {name}, {module.qweight.dtype}")
        if module.qweight.numel() != module.in_features * module.out_features // 2:
            raise RuntimeError(f"Unpacked weight storage: {name}")
        if module.activation_bits != int(method[-1]) or module.weight_bits != 4:
            raise RuntimeError(f"Incorrect precision: {name}")
        if module.stream_gains.shape[0] != 5 or module.stream_gains.shape[1] not in (1, 11):
            raise RuntimeError(f"Incorrect RoboCasa stream schedule: {name}")
        if method == "w4a4" and not module._fuses_input_rotation:
            raise RuntimeError(f"W4A4 rotation is not fused: {name}")
        for attr in ("_fast_wam_runner", "_fast_adaln_runner", "_fast_gate_residual_runner"):
            fn = getattr(module, attr)
            if fn is None:
                raise RuntimeError(f"Missing native kernel: {name}.{attr}")

            def counted(*args, _fn=fn, _name=name, **kwargs):
                calls[_name] += 1
                return _fn(*args, **kwargs)

            setattr(module, attr, counted)
    return dict(
        module_count=len(modules),
        backend="cutlass",
        weight_bits=4,
        activation_bits=int(method[-1]),
        stream_counts=dict(Counter(str(m.stream_gains.shape[1]) for _, m in modules)),
        rotation_fused=method == "w4a4",
        fallback="forbidden",
    ), calls


def array_hash(value):
    import numpy as np

    arr = np.ascontiguousarray(value)
    h = hashlib.sha256(str((arr.dtype.str, arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def create_episode(evaluation, cfg, episode, root):
    from cosmos_policy.utils.utils import set_seed_everywhere

    cfg.task_name = episode["task"]
    cfg.seed = episode["policy_seed"]
    cfg.deterministic_reset_seed = episode["env_seed"]
    # Model loading / prior rollout RNG consumption cannot affect initialization.
    set_seed_everywhere(episode["env_seed"])
    env, kwargs = evaluation.create_robocasa_env(
        cfg, seed=episode["env_seed"], episode_idx=episode["episode_id"]
    )
    try:
        if list(kwargs["layout_and_style_ids"][0]) != episode["layout_style"]:
            raise RuntimeError("Scene selection differs from fixed manifest")
        set_seed_everywhere(episode["env_seed"])
        obs = env.reset()  # Exactly one reset, matching the official rollout path.
        meta = env.get_ep_meta()
        if [meta["layout_id"], meta["style_id"]] != episode["layout_style"]:
            raise RuntimeError("Actual environment scene differs from manifest")
        state = env.sim.get_state().flatten()
        xml = env.sim.model.get_xml()
        fingerprint = dict(
            episode=episode,
            model_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            state_sha256=array_hash(state),
            metadata=normalize_metadata(meta),
            metadata_sha256=digest(normalize_metadata(meta)),
            observations={k: array_hash(v) for k, v in sorted(obs.items())},
        )
        shared_json(root / "initial_states" / f"{episode['key']}.json", fingerprint)
        print(f"[initial-state-verified] {episode['key']} state={fingerprint['state_sha256']}", flush=True)
        return env, obs, meta["lang"], digest(fingerprint)
    except BaseException:
        env.close()
        raise


def checked_action(original, model, method, *args, **kwargs):
    import numpy as np

    tracker = getattr(model.net, "_wam_network_call_tracker", None)
    before = tracker.total_calls if tracker is not None else None
    result = original(*args, **kwargs)
    actions = np.asarray(result["actions"])
    if actions.shape != (32, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid action chunk: {actions.shape}")
    if tracker is not None:
        if tracker.total_calls - before != 5:
            raise RuntimeError("Action query did not execute exactly five denoising calls")
        tracker.validate_complete_schedule()
    return result


def smoke(evaluation, cfg, model, stats, method, root, identity, audit, counts):
    import numpy as np
    import torch

    manifest = make_manifest()
    cases = [
        e
        for e in manifest["episodes"]
        if e["task"] == "TurnOffMicrowave" and e["policy_seed"] == 195 and e["episode_id"] % 10 == 0
    ]
    checks = []
    kernels = Counter()
    for case_index, episode in enumerate(cases):
        env = None
        try:
            env, obs, language, state_hash = create_episode(evaluation, cfg, episode, root)
            for _ in range(10):
                obs, _, _, _ = env.step(np.zeros(env.action_dim))
            for query in range(2):
                observation = evaluation.prepare_observation(obs, cfg.flip_images)

                def call():
                    return evaluation.get_action(
                        cfg,
                        model,
                        stats,
                        copy.deepcopy(observation),
                        language,
                        seed=cfg.seed,
                        randomize_seed=False,
                        num_denoising_steps_action=5,
                        generate_future_state_and_value_in_parallel=True,
                    )

                if case_index == 0 and query == 1:
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    ) as prof:
                        result = call()
                        torch.cuda.synchronize()
                    for event in prof.events():
                        if event.device_type == torch.autograd.DeviceType.CUDA:
                            kernels[event.name] += 1
                else:
                    result = call()
                for action in result["actions"][:16]:
                    if env.action_dim == 12:
                        action = np.concatenate((action, [0, 0, 0, 0, -1]))
                    obs, _, _, _ = env.step(action)
            checks.append(
                dict(
                    key=episode["key"],
                    scene=episode["layout_style"],
                    initial_state_sha256=state_hash,
                    action_queries=2,
                    simulation_steps=42,
                )
            )
            print(f"[smoke] {method} scene={episode['scene_index']} passed", flush=True)
        finally:
            if env is not None:
                env.close()
    integer_kernels = {
        k: v
        for k, v in kernels.items()
        if "cutlass" in k.lower() and any(t in k.lower() for t in ("integer_subbyte", "int4b"))
    }
    if method != "fp" and (len(counts) != 280 or not integer_kernels):
        raise RuntimeError(
            f"Native backend not established: {len(counts)} modules, {len(integer_kernels)} INT4 kernels"
        )
    expected_graphs = (1 if method == "fp" else 5) if cfg.dit_cuda_graph else 0
    if getattr(model.net, "captured_graph_count", 0) != expected_graphs:
        raise RuntimeError(f"Unexpected CUDA Graph count: {getattr(model.net, 'captured_graph_count', 0)}")
    # Exercise the exact official rollout function used by formal evaluation.
    # This separate trial remains smoke-only, even if it succeeds.
    env = None
    try:
        env, _, language, state_hash = create_episode(evaluation, cfg, cases[0], root)
        rollout = evaluation.run_episode(cfg, env, language, model, None, stats, None, cases[0]["episode_id"])
        full_rollout = dict(
            key=cases[0]["key"],
            success=bool(rollout[0]),
            length=int(rollout[1]),
            initial_state_sha256=state_hash,
            included_in_formal_results=False,
        )
        del rollout
    finally:
        if env is not None:
            env.close()
    report = dict(
        passed=True,
        identity=identity,
        backend=audit,
        checks=checks,
        full_rollout=full_rollout,
        captured_graphs=getattr(model.net, "captured_graph_count", 0),
        kernel_calls=dict(counts),
        integer_cuda_kernels=integer_kernels,
        cuda_kernels=dict(kernels),
        note="Five scenes with two action queries each, plus one official full rollout; smoke only.",
        completed_unix=time.time(),
    )
    write_json(root / "smoke" / f"{method}.json", report)
    print(f"[smoke-passed] {method}: {root / 'smoke' / (method + '.json')}", flush=True)


def run(evaluation, cfg, model, stats, method, root, identity, manifest):
    results_dir = root / method / "episodes"
    results_dir.mkdir(parents=True, exist_ok=True)
    shared_json(root / method / "identity.json", identity)
    completed = []
    identity_hash = digest(identity)
    for episode in manifest["episodes"]:
        path = results_dir / f"{episode['key']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity_sha256"] != identity_hash:
                raise RuntimeError(f"Result provenance changed: {path}")
            fingerprint = json.loads((root / "initial_states" / path.name).read_text())
            if result["initial_state_sha256"] != digest(fingerprint):
                raise RuntimeError(f"Initial state record changed: {path}")
        else:
            env = None
            started = time.monotonic()
            try:
                env, _, language, state_hash = create_episode(evaluation, cfg, episode, root)
                output = evaluation.run_episode(
                    cfg, env, language, model, None, stats, None, episode["episode_id"]
                )
                result = dict(
                    **episode,
                    success=bool(output[0]),
                    length=int(output[1]),
                    seconds=time.monotonic() - started,
                    identity_sha256=identity_hash,
                    initial_state_sha256=state_hash,
                    method=method,
                )
                del output
                write_json(path, result)
            finally:
                if env is not None:
                    env.close()
        completed.append(result)
        summary = summarize(manifest, completed)
        write_json(root / method / "summary.json", summary)
        print(f"[progress] {method} {len(completed)}/720 successes={summary['successes']}", flush=True)


def main():
    args = validate_args(build_parser().parse_args())
    if not args.check_only and os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Start Python with PYTHONHASHSEED=0 (see docs/robocasa.md)")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from .integrations.cosmos import prepare_paths

    prepare_paths(args)
    if args.check_only:
        print(
            json.dumps(
                dict(
                    check_only_passed=True,
                    method=args.method,
                    suite=args.suite,
                    model=str(args.checkpoint),
                    quantization=str(args.quant_checkpoint),
                    cuda_graph=args.cuda_graph,
                    expected_episodes=720,
                ),
                indent=2,
            )
        )
        return
    import torch
    from cosmos_policy.constants import ROBOT_PLATFORM, NUM_ACTIONS_CHUNK
    from cosmos_policy.experiments.robot.robocasa import run_robocasa_eval as evaluation

    # The pinned upstream evaluator otherwise resolves this relative to cwd.
    controller_path = Path(evaluation.__file__).with_name("robocasa_controller_configs.pkl")
    if not controller_path.is_file():
        raise FileNotFoundError(f"Missing installed RoboCasa controller asset: {controller_path}")
    evaluation.CONTROLLER_CONFIGS_PATH = str(controller_path)
    from .cuda_graph import CUDAGraphInferenceWrapper
    from cosmos_policy.utils.utils import set_seed_everywhere

    if ROBOT_PLATFORM != "ROBOCASA" or NUM_ACTIONS_CHUNK != 32:
        raise RuntimeError("RoboCasa environment variables must be set before import")
    if set(TASKS) != set(evaluation.TASK_MAX_STEPS):
        raise RuntimeError("Official task list changed")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / f"{args.method}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = make_manifest()
    shared_json(root / "manifest.json", manifest)
    cfg = make_config(args, evaluation)
    identity = provenance(args, manifest, evaluation)
    identity["config"] = {
        **asdict(cfg),
        "dit_cuda_graph": args.cuda_graph,
        "dit_cuda_graph_warmup": 3,
        "method": args.method,
        "quantization_backend": "cutlass" if cfg.quantization_enabled else None,
    }
    report_path = root / "smoke" / f"{args.method}.json"
    if args.mode in ("run", "check"):
        if not report_path.is_file():
            raise RuntimeError("Run --mode smoke first; backend smoke report is missing")
        report = json.loads(report_path.read_text())
        if not report.get("passed") or report["identity"] != identity:
            raise RuntimeError("Smoke is stale or failed; rerun --mode smoke")
        if args.mode == "check":
            print(f"[smoke-current] {args.method}", flush=True)
            return
    set_seed_everywhere(195)
    evaluation.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = evaluation.load_dataset_stats(cfg.dataset_stats_path)
    model, model_cfg = load_model(cfg)
    if model_cfg.dataloader_train.dataset.chunk_size != 32:
        raise RuntimeError("Checkpoint chunk size mismatch")
    audit, counts = install_kernel_guards(model, args.method)
    if args.cuda_graph:
        model.net = CUDAGraphInferenceWrapper(model.net, warmup=3).eval()
    cfg.dit_cuda_graph = args.cuda_graph
    original_get_action = evaluation.get_action
    evaluation.get_action = lambda *a, **kw: checked_action(original_get_action, model, args.method, *a, **kw)
    try:
        with torch.inference_mode():
            if args.mode == "smoke":
                smoke(evaluation, cfg, model, stats, args.method, root, identity, audit, counts)
            else:
                run(evaluation, cfg, model, stats, args.method, root, identity, manifest)
    finally:
        evaluation.get_action = original_get_action
        lock.close()


if __name__ == "__main__":
    main()
