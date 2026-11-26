"""Calibrate a WAM checkpoint for Cosmos-Policy LIBERO inference."""

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import time
from typing import Any
import torch
from steerquant.integrations.cosmos import get_model, init_t5_text_embeddings_cache, load_dataset_stats
from steerquant import (
    WAMActivationCache,
    WAMActivationCacheCollector,
    WAMCacheConfig,
    WAMDCalibrationConfig,
    WAMGammaCalibrationConfig,
    build_wam_cache_sites,
    calibrate_wam_model,
    load_wam_sensitivity_field_csv,
)
from steerquant.integrations.calibration_data import (
    build_eval_config,
    build_policy_data_batch,
    load_calibration,
)
from steerquant.integrations.calibration_data import generate_matched
from steerquant.integrations.calibration_data import resolve_checkpoint
from steerquant.wam_pipeline import calibrate_wam_model_grouped, WAMLinearCheckpoint


class _DenoiseCallBridge:
    """Tell the cache collector which denoise-call lookup is being executed."""

    def __init__(
        self, net: torch.nn.Module, collector: WAMActivationCacheCollector, num_denoise_calls: int
    ) -> None:
        self.net = net
        self.collector = collector
        self.num_denoise_calls = int(num_denoise_calls)
        self.observation_calls = 0
        self.total_calls = 0
        self._handle: Any | None = None

    def _pre_hook(self, _module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        if self.observation_calls >= self.num_denoise_calls:
            raise RuntimeError(
                "The denoiser exceeded the configured number of WAM calls for one observation."
            )
        self.collector.begin_denoise_call(self.observation_calls)
        self.observation_calls += 1
        self.total_calls += 1

    def register(self) -> None:
        if self._handle is not None:
            raise RuntimeError("WAM denoise-call bridge is already registered.")
        self._handle = self.net.register_forward_pre_hook(self._pre_hook)

    def begin_observation(self, observation_id: int) -> None:
        if not self.collector.begin_observation(observation_id):
            raise RuntimeError(f"WAM cache rejected calibration observation {observation_id}.")
        self.observation_calls = 0

    def finish_observation(self) -> None:
        if self.observation_calls != self.num_denoise_calls:
            raise RuntimeError(
                f"Expected {self.num_denoise_calls} denoiser calls, observed {self.observation_calls}."
            )

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("WAM calibration requires a CUDA GPU.")
    if args.observation_start < 0 or args.max_observations <= 0:
        raise ValueError("Observation start must be non-negative and count must be positive.")
    if args.expected_sensitivity_observations <= 0:
        raise ValueError("Expected sensitivity observation count must be positive.")
    if args.num_denoising_steps != 5:
        raise ValueError("This WAM sensitivity field and LIBERO checkpoint require five denoise calls.")
    expected_chunk, expected_streams = {"libero": (16, 9), "robocasa": (32, 11)}[getattr(args, "suite", "libero")]
    if args.chunk_size != expected_chunk or args.num_streams != expected_streams:
        raise ValueError(f"Selected suite requires chunk_size={expected_chunk}, num_streams={expected_streams}.")
    if args.tokens_per_observation <= 0:
        raise ValueError("tokens-per-observation must be positive.")
    if args.train_rows_per_cell < 0:
        raise ValueError("train-rows-per-cell must be non-negative; use zero for exact training.")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive.")
    output = args.output.expanduser().resolve()
    if output.exists() and (not args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing WAM checkpoint: {output}")


def _collect_cache(
    args: argparse.Namespace,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    dataset_stats: dict[str, Any],
    eval_cfg: Any,
    sensitivity_field: Any,
    layers: tuple[int, ...] | None = None,
) -> WAMActivationCache:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=len(sensitivity_field.module_names),
            num_denoise_calls=args.num_denoising_steps,
            num_streams=args.num_streams,
            max_observations=len(records),
            tokens_per_observation=args.tokens_per_observation,
            seed=args.sampling_seed,
            storage_dtype=args.storage_dtype,
        )
    )
    sites = build_wam_cache_sites(model.net, sensitivity_field)
    if layers is not None:
        selected = set(layers)
        sites = [site for site in sites if site.layer in selected]
        if not sites or len(sites) != len(layers):
            raise ValueError("Invalid reconstruction cache layer selection")
    collector = WAMActivationCacheCollector(cache, sites)
    bridge = _DenoiseCallBridge(model.net, collector, args.num_denoising_steps)
    collector.register()
    bridge.register()
    started = time.perf_counter()
    try:
        for local_index, record in enumerate(records):
            observation_id = args.observation_start + local_index
            data_batch, _ = build_policy_data_batch(
                eval_cfg, model, dataset_stats, record["observation"], record["task_label"]
            )
            bridge.begin_observation(observation_id)
            with torch.no_grad():
                generated = generate_matched(
                    model, data_batch, seed=args.seed, num_steps=args.num_denoising_steps
                )
            bridge.finish_observation()
            del generated, data_batch
            completed = local_index + 1
            if completed % args.progress_every == 0 or completed == len(records):
                elapsed = time.perf_counter() - started
                print(
                    f"[cache] {completed}/{len(records)} observations, elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
            if completed % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        bridge.remove()
        collector.remove()
    mask = sensitivity_field.active_stream_mask
    expected_cells = int((mask if layers is None else mask[list(layers)]).sum().item())
    if len(cache.keys) != expected_cells:
        raise RuntimeError(f"Collected {len(cache.keys)} WAM cells, expected {expected_cells}.")
    if len(cache.accepted_observations) != len(records):
        raise RuntimeError("WAM cache did not retain every requested calibration observation.")
    print(
        f"[cache] complete: cells={len(cache.keys)}, tensor_storage={cache.storage_nbytes / 2**30:.2f} GiB, network_calls={bridge.total_calls}",
        flush=True,
    )
    return cache


def run(args) -> None:
    _validate_args(args)
    torch.set_float32_matmul_precision("high")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    tokenizer_path = args.tokenizer_checkpoint.expanduser().resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    os.environ["COSMOS_POLICY_CKPT"] = str(checkpoint_path)
    os.environ["COSMOS_POLICY_TOKENIZER"] = str(tokenizer_path)
    print("[wam] loading observation-level sensitivity field", flush=True)
    sensitivity_field = load_wam_sensitivity_field_csv(
        args.sensitivity,
        num_denoise_calls=args.num_denoising_steps,
        num_semantic_streams=args.num_streams,
        expected_observations=args.expected_sensitivity_observations,
    )
    if len(sensitivity_field.module_names) != 280:
        raise RuntimeError(f"Expected 280 WAM Linear sites, found {len(sensitivity_field.module_names)}.")
    dataset_stats = load_dataset_stats(str(args.dataset_stats.expanduser().resolve()))
    init_t5_text_embeddings_cache(str(args.t5_embeddings.expanduser().resolve()))
    eval_cfg = build_eval_config(args, checkpoint_path)
    model, _ = get_model(eval_cfg)
    model.eval().requires_grad_(False)
    rotations = None
    if args.rotation_enabled:
        from .algorithms.preprocess import apply_wam_rotations_

        rotations = apply_wam_rotations_(model.net, args.rotation_seed)
        if len(rotations) != 280:
            raise RuntimeError(f"Expected 280 rotated Linear sites, found {len(rotations)}")
        print(f"[rotation] installed {len(rotations)} local Hadamard rotations", flush=True)
    all_records = load_calibration(args.calibration, None, suite=getattr(args, "suite", "libero"))
    records = all_records[args.observation_start : args.observation_start + args.max_observations]
    if len(records) != args.max_observations:
        raise ValueError("Requested observation range exceeds the calibration set.")
    group_size = getattr(args, "cache_layers_per_group", 0)
    cache = None if group_size else _collect_cache(args, model, records, dataset_stats, eval_cfg, sensitivity_field)
    train_rows = None if args.train_rows_per_cell == 0 else args.train_rows_per_cell
    d_config = WAMDCalibrationConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        epochs=args.d_epochs,
        learning_rate=args.d_learning_rate,
        batch_size=args.batch_size,
        smoothquant_alpha=args.smoothquant_alpha,
        hard_eval_interval=args.hard_eval_interval,
        train_rows_per_cell=train_rows,
        sampling_seed=args.sampling_seed,
    )
    gamma_config = WAMGammaCalibrationConfig(
        activation_bits=args.activation_bits,
        epochs=args.gamma_epochs,
        gain_learning_rate=args.gain_learning_rate,
        clip_learning_rate=args.clip_learning_rate,
        batch_size=args.batch_size,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        hard_eval_interval=args.hard_eval_interval,
        train_rows_per_cell=train_rows,
        sampling_seed=args.sampling_seed + 1,
    )
    calibration_started = time.perf_counter()

    def report_progress(layer: int, total: int, module_name: str) -> None:
        if layer:
            gc.collect()
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - calibration_started
        print(
            f"[optimize] layer={layer + 1}/{total} module={module_name} elapsed={elapsed / 3600:.2f} h",
            flush=True,
        )

    common = dict(
        sensitivity_field=sensitivity_field,
        rho=args.rho,
        eps=args.epsilon,
        u_min=args.u_min,
        u_max=args.u_max,
        d_config=d_config,
        gamma_config=gamma_config,
        progress=report_progress,
        input_rotation_signs=rotations,
    )
    if group_size:
        from pathlib import Path
        shard_dir = args.output.parent / 'optimized_layers'
        shard_dir.mkdir(exist_ok=True)
        # Bind saved optimizer results to exact sensitivity/config inputs. The
        # group size changes only the lifetime of identical sampled tensors.
        digest = hashlib.sha256()
        for source in [args.output.parent / 'config.json', args.sensitivity]:
            with Path(source).open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(block)
        contract = digest.hexdigest()
        completed = {}
        for path in sorted(shard_dir.glob('*.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if payload.get('contract') != contract:
                raise ValueError(f'Optimizer resume contract mismatch: {path}')
            entry = WAMLinearCheckpoint(**payload['entry'])
            if path.stem != f'{entry.layer:04d}' or entry.layer in completed:
                raise ValueError(f'Invalid optimizer shard identity: {path}')
            completed[entry.layer] = entry
        print(f'[optimize-resume] restored {len(completed)}/280 layers', flush=True)

        def save_layer(entry):
            from dataclasses import asdict
            temporary = shard_dir / f'{entry.layer:04d}.tmp'
            with temporary.open('wb') as handle:
                torch.save({'contract': contract, 'entry': asdict(entry)}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(shard_dir / f'{entry.layer:04d}.pt')
            print(f'[optimize-saved] layer={entry.layer + 1}/280', flush=True)

        def provide(layers):
            gc.collect()
            torch.cuda.empty_cache()
            print(f'[cache-group] layers={layers[0] + 1}-{layers[-1] + 1}/280 count={len(layers)}', flush=True)
            return _collect_cache(args, model, records, dataset_stats, eval_cfg, sensitivity_field, layers)

        wam_checkpoint = calibrate_wam_model_grouped(
            model.net, cache_provider=provide, layers_per_group=group_size,
            completed_linears=completed, layer_completed=save_layer, **common,
        )
    else:
        wam_checkpoint = calibrate_wam_model(
            model.net, cache=cache, release_cache_after_layer=True, **common,
        )
    output = wam_checkpoint.save(args.output)
    total_elapsed = time.perf_counter() - calibration_started
    summary = {
        "suite": getattr(args, "suite", "libero"),
        "chunk_size": args.chunk_size,
        "num_semantic_streams": args.num_streams,
        "format": "cosmos_policy_wam_calibration_run_v1",
        "output": str(output),
        "base_checkpoint": str(checkpoint_path),
        "sensitivity": str(args.sensitivity.expanduser().resolve()),
        "calibration": str(args.calibration.expanduser().resolve()),
        "observation_start": args.observation_start,
        "observation_count": len(records),
        "tokens_per_observation": args.tokens_per_observation,
        "train_rows_per_cell": train_rows,
        "stochastic_training": train_rows is not None,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "rho": args.rho,
        "epsilon": args.epsilon,
        "sampling_seed": args.sampling_seed,
        "d_config": wam_checkpoint.d_config,
        "gamma_config": wam_checkpoint.gamma_config,
        "linear_count": len(wam_checkpoint.linears),
        "rotation_enabled": args.rotation_enabled,
        "rotation_seed": args.rotation_seed if args.rotation_enabled else None,
        "rotation_count": len(rotations) if rotations else 0,
        "calibration_seconds": total_elapsed,
        "cache_layers_per_group": group_size,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
