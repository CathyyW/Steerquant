"""Complete calibration: reference ranges -> action sensitivity -> D/gamma."""

from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("libero", "robocasa"), default="libero")
    p.add_argument("--check-only", action="store_true", help="Validate inputs without model loading or calibration")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    p.add_argument("--rotation-seed", type=int, default=20260904, help="W4A4 randomized Hadamard seed")
    p.add_argument("--observations", type=int, default=None,
                   help="Calibration observations (default: LIBERO 320, RoboCasa 192)")
    p.add_argument("--action-probes", type=int, default=16)
    p.add_argument("--tokens-per-observation", type=int, default=2)
    p.add_argument("--train-rows-per-cell", type=int, default=512)
    p.add_argument("--d-epochs", type=int, default=20)
    p.add_argument("--gamma-epochs", type=int, default=30)
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=1e-16)
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--probe-seed", type=int, default=20260809)
    p.add_argument("--sensitivity-memory", choices=("checkpoint", "cpu"), default="checkpoint")
    p.add_argument("--sensitivity-only", action="store_true")
    p.add_argument("--reuse-ranges", type=Path, help="Reuse activation_stats.pt with matching adjacent config.json")
    p.add_argument("--resume", action="store_true", help="Resume completed sensitivity observations with identical configuration")
    p.add_argument("--cuda-memory-fraction", type=float, default=None, help="Optional per-process CUDA allocator limit for concurrent calibration")
    p.add_argument("--sampling-seed", type=int, default=20260831)
    p.add_argument("--cache-layers-per-group", type=int, default=0,
                   help="Bound reconstruction cache to this many Linear sites; zero caches all sites")
    return p


def collect_ranges(model, records, cfg, stats, seed, rotation_seed=None):
    import torch
    from .integrations.linear_sites import enumerate_block_linears, NetworkCallTracker
    from .integrations.calibration_data import build_policy_data_batch, generate_matched

    tracker = NetworkCallTracker(model.net, 5)
    ranges, handles = {}, []

    from .algorithms.preprocess import preprocess_signs, randomized_hadamard_transform

    def hook(name, signs):
        def collect(module, inputs):
            x = inputs[0].detach()
            if signs is not None:
                x = randomized_hadamard_transform(x, signs)
            x = x.float().reshape(-1, x.shape[-1])
            maximum = x.abs().amax(0).cpu()
            if name not in ranges:
                ranges[name] = torch.zeros(5, maximum.numel())
            call = tracker.current_call
            ranges[name][call] = torch.maximum(ranges[name][call], maximum)

        return collect

    tracker.register()
    for site in enumerate_block_linears(model.net):
        name = f"blocks.{site.block_index - 1}.{site.module_path}"
        signs = (
            None
            if rotation_seed is None
            else preprocess_signs(
                site.module.in_features,
                seed=rotation_seed,
                module_name=name,
                device=site.module.weight.device,
            )
        )
        handles.append(site.module.register_forward_pre_hook(hook(name, signs)))
    try:
        for index, record in enumerate(records):
            tracker.reset()
            batch, _ = build_policy_data_batch(cfg, model, stats, record["observation"], record["task_label"])
            generate_matched(model, batch, seed=seed, num_steps=5)
            if tracker.total_calls != 5:
                raise RuntimeError(f"Expected five denoise calls, got {tracker.total_calls}")
            print(f"[ranges] {index + 1}/{len(records)}", flush=True)
    finally:
        tracker.remove()
        for handle in handles:
            handle.remove()
    return ranges


def collect_sensitivity(model, records, cfg, stats, ranges, args):
    import torch
    from cosmos_policy.constants import ACTION_DIM
    from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
    from .integrations.calibration_data import (
        build_policy_data_batch,
        differentiable_sample,
        unnormalize_actions_torch,
    )
    from .sensitivity import FirstOrderLinearCollector
    from .map_reuse import retained_backward_state, run_action_probes
    from .integrations.linear_sites import SLOT_NAMES, ROBOCASA_SLOT_NAMES

    collector = FirstOrderLinearCollector(
        model.net,
        weight_bits=4,
        activation_bits=args.activation_bits,
        activation_granularity="per_tensor",
        activation_clipping_ranges=ranges,
        num_action_probes=args.action_probes,
        num_action_elements=args.chunk_size * ACTION_DIM,
        num_denoising_steps=5,
        rotation_seed=args.rotation_seed if args.rotation_enabled else None,
        slot_names=ROBOCASA_SLOT_NAMES if args.suite == "robocasa" else SLOT_NAMES,
        checkpoint_denoisers=args.sensitivity_memory == "checkpoint",
        reuse_forward=True,
    )
    action_std = torch.as_tensor(stats["actions_std"], device="cuda", dtype=torch.float32)
    collector.register()
    output = args.output_dir / "sensitivity.csv"
    shards = args.output_dir / "sensitivity_observations"
    shards.mkdir(exist_ok=True)
    expected_keys = {
        (site.block_index, site.linear_index, call, slot)
        for site in collector.sites for call in range(5)
        for slot in (range(len(collector.slot_names)) if site.slot_aligned else (-1,))
    }
    try:
        with output.open("w", newline="") as handle:
            writer = None
            for index, record in enumerate(records):
                shard = shards / f"{index:04d}.json"
                if shard.exists():
                    if not args.resume:
                        raise FileExistsError(f"Refusing existing sensitivity shard: {shard}")
                    rows = json.loads(shard.read_text())
                    keys = {(r['block_index'], r['linear_index'], r['denoise_call'], r['slot_index']) for r in rows}
                    if len(rows) != len(expected_keys) or keys != expected_keys or any(
                        r['observation_id'] != record['observation_id'] or
                        not math.isfinite(r['predicted_standardized_action_rmse']) or
                        r['predicted_standardized_action_rmse'] < 0 for r in rows
                    ):
                        raise ValueError(f"Invalid sensitivity shard: {shard}")
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
                    handle.flush()
                    print(f"[sensitivity] restored {index + 1}/{len(records)}", flush=True)
                    continue
                collector.reset_observation()
                generator = torch.Generator(device="cuda").manual_seed(args.probe_seed + index)
                torch.cuda.reset_peak_memory_stats()

                def build_forward():
                    batch, action_index = build_policy_data_batch(
                        cfg, model, stats, record["observation"], record["task_label"]
                    )
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed_all(args.seed)
                    context = nullcontext() if collector.checkpoint_denoisers else torch.autograd.graph.save_on_cpu(pin_memory=False)
                    with context:
                        generated = differentiable_sample(
                            model, batch, collector, seed=args.seed, num_steps=5
                        )
                    normalized = extract_action_chunk_from_latent_sequence(
                        generated,
                        action_shape=(args.chunk_size, ACTION_DIM),
                        action_indices=torch.tensor([action_index], device=generated.device),
                    ).float()
                    return unnormalize_actions_torch(normalized, stats) / (action_std + 1e-8)

                def report_probe(probe, seconds):
                    print(f"[probe] observation={index + 1}/{len(records)} probe={probe + 1}/{args.action_probes} seconds={seconds:.1f} gpu_peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)

                with retained_backward_state(
                    model.net, enabled=not collector.checkpoint_denoisers
                ):
                    forwards = run_action_probes(
                        build_forward, collector, generator,
                        on_probe=report_probe,
                    )
                print(f"[map] observation={index + 1}/{len(records)} sampling_forwards={forwards} reverse_passes={args.action_probes}", flush=True)
                rows = collector.export_action_probe_rows({"observation_id": record["observation_id"]})
                temporary = shard.with_suffix('.tmp')
                with temporary.open('w') as checkpoint_file:
                    json.dump(rows, checkpoint_file, allow_nan=False)
                    checkpoint_file.flush()
                    os.fsync(checkpoint_file.fileno())
                temporary.replace(shard)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                print(f"[sensitivity] {index + 1}/{len(records)}", flush=True)
    finally:
        collector.remove()
    return output


def resolve_defaults(args):
    if args.observations is None:
        args.observations = 192 if args.suite == "robocasa" else 320
    return args


def main():
    args = resolve_defaults(build_parser().parse_args())
    # Tokenizer determinism alone does not control attention backward. Cover
    # both PyTorch SDPA (RoboCasa's minimal_a2a backend) and Transformer Engine.
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    args.attention_deterministic = True
    args.deterministic_algorithms = True
    args.rotation_enabled = args.activation_bits == 4
    if args.observations < 1 or args.action_probes < 1 or args.d_epochs < 0 or args.gamma_epochs < 0:
        raise ValueError("Observation/probe counts must be positive; epochs must be nonnegative")
    if args.cache_layers_per_group < 0:
        raise ValueError("cache-layers-per-group must be nonnegative")
    # Refuse to combine old partial results with a new calibration run.
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Use a new or empty --output-dir for each calibration")
    if args.resume and not (args.output_dir / 'config.json').is_file():
        raise FileNotFoundError("Resume requires the original config.json")
    if (args.output_dir / f'steerquant_w4a{args.activation_bits}.pt').exists():
        raise FileExistsError("Calibration already has a final checkpoint")
    from .integrations.cosmos import (
        prepare_paths,
        get_model,
        load_dataset_stats,
        init_t5_text_embeddings_cache,
    )
    prepare_paths(args)
    from .integrations.calibration_data import load_calibration, build_eval_config
    from .integrations.linear_sites import unwrap_checkpointed_blocks
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.__dict__.update(
        map_forward_mode="reuse",
        weight_bits=4,
        num_denoising_steps=5,
        chunk_size=32 if args.suite == "robocasa" else 16,
        num_streams=11 if args.suite == "robocasa" else 9,
        # Match the existing RoboCasa FP baseline collection and official inference.
        recompress_jpeg=args.suite == "robocasa",
        observation_start=0,
        max_observations=args.observations,
        expected_sensitivity_observations=args.observations,
        storage_dtype="float16",
        batch_size=512,
        hard_eval_interval=5,
        smoothquant_alpha=0.5,
        d_learning_rate=0.02,
        gain_learning_rate=0.02,
        clip_learning_rate=0.01,
        gamma_min=0.25,
        gamma_max=4.0,
        u_min=None,
        u_max=None,
        progress_every=10,
        overwrite=False,
    )
    from cosmos_policy.constants import ROBOT_PLATFORM, NUM_ACTIONS_CHUNK
    if ROBOT_PLATFORM.lower() != args.suite or NUM_ACTIONS_CHUNK != args.chunk_size:
        raise RuntimeError("Cosmos platform constants disagree with --suite; check command-line paths")
    records = load_calibration(args.calibration, args.observations, suite=args.suite)
    if len(records) != args.observations:
        raise ValueError("Calibration file contains fewer observations than requested")
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with args.calibration.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    metadata["calibration_sha256"] = digest.hexdigest()
    if args.resume:
        previous = json.loads((args.output_dir / 'config.json').read_text())
        ignored = {'resume', 'check_only', 'sensitivity_only', 'cache_layers_per_group'}
        mismatches = [k for k in set(previous) | set(metadata)
                      if k not in ignored and previous.get(k) != metadata.get(k)]
        if mismatches:
            raise ValueError(f"Resume configuration mismatch: {mismatches}")
    if args.check_only:
        print(json.dumps({"check_only_passed": True, **metadata}, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("SteerQuant calibration requires CUDA")
    if args.cuda_memory_fraction is not None:
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError('cuda-memory-fraction must be in (0, 1]')
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stats = load_dataset_stats(str(args.dataset_stats))
    init_t5_text_embeddings_cache(str(args.t5_embeddings))
    cfg = build_eval_config(args, args.checkpoint)
    model, _ = get_model(cfg)
    model.eval().requires_grad_(False)
    unwrap_checkpointed_blocks(model.net)
    range_path = args.output_dir / 'activation_stats.pt' if args.resume else args.reuse_ranges
    if range_path is not None and range_path.is_file():
        range_metadata = json.loads((range_path.parent / 'config.json').read_text())
        range_keys = ('suite', 'calibration_sha256', 'checkpoint', 'tokenizer_checkpoint',
                      'observations', 'seed', 'rotation_enabled', 'rotation_seed',
                      'num_denoising_steps', 'recompress_jpeg', 'config', 'config_file',
                      'dataset_stats', 't5_embeddings')
        mismatches = [k for k in range_keys if range_metadata.get(k) != metadata.get(k)]
        if mismatches:
            raise ValueError(f"Activation range configuration mismatch: {mismatches}")
        loaded = torch.load(range_path, map_location='cpu', weights_only=True)
        from .integrations.linear_sites import enumerate_block_linears
        sites = enumerate_block_linears(model.net)
        expected = {f'blocks.{s.block_index - 1}.{s.module_path}': s.module.in_features for s in sites}
        if set(loaded) != set(expected):
            raise ValueError('Activation range Linear sites mismatch')
        ranges = {name: item['timestep_absmax'] for name, item in loaded.items()}
        for name, value in ranges.items():
            if value.shape != (5, expected[name]) or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f'Invalid activation range: {name}')
        print(f'[ranges] reused {range_path}', flush=True)
    else:
        if args.reuse_ranges is not None and not args.resume:
            raise FileNotFoundError(args.reuse_ranges)
        ranges = collect_ranges(
            model, records, cfg, stats, args.seed,
            rotation_seed=args.rotation_seed if args.rotation_enabled else None,
        )
    range_temporary = args.output_dir / 'activation_stats.pt.tmp'
    torch.save(
        {name: {"timestep_absmax": value} for name, value in ranges.items()},
        range_temporary,
    )
    range_temporary.replace(args.output_dir / 'activation_stats.pt')
    args.sensitivity = collect_sensitivity(
        model, records, cfg, stats, {name: value.amax(1) for name, value in ranges.items()}, args
    )
    if args.sensitivity_only:
        return
    del model, records, ranges
    gc.collect()
    torch.cuda.empty_cache()
    args.output = args.output_dir / f"steerquant_w4a{args.activation_bits}.pt"
    from .calibration import run

    run(args)


if __name__ == "__main__":
    main()
