#!/usr/bin/env python3
"""Build a task- and progress-balanced LIBERO calibration set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


PHASE_NAMES = ("early", "early_middle", "late_middle", "late")
FORMAT_VERSION = "cosmos_policy.libero.calibration_window.v1"


def parse_args() -> argparse.Namespace:
    workspace = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=workspace / "LIBERO-Cosmos-Policy" / "success_only",
        help="Directory containing the four *_regen task-suite directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "calibration",
        help="Destination directory.",
    )
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--demos-per-task", type=int, default=8)
    parser.add_argument("--expected-tasks", type=int, default=40)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated sample files and manifests.",
    )
    return parser.parse_args()


def task_seed(seed: int, suite: str, task: str) -> int:
    """Return a stable task-local seed independent of filesystem ordering."""
    payload = f"{seed}\0{suite}\0{task}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def suite_name(path: Path) -> str:
    name = path.parent.name
    return name.removesuffix("_regen")


def task_label(path: Path) -> str:
    """Match LIBERODataset's filename-to-language conversion."""
    filename = path.name
    if not filename.endswith("_demo.hdf5"):
        raise ValueError(f"Unexpected LIBERO task filename: {filename}")
    words = filename[: -len("_demo.hdf5")].split("_")
    scene_indices = [index for index, word in enumerate(words) if "SCENE" in word]
    if scene_indices:
        words = words[scene_indices[-1] + 1 :]
    return " ".join(words)


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = list(data_group.keys())
    try:
        return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected demo keys: {keys[:5]}") from exc


def frame_count(demo: h5py.Group) -> int:
    obs = demo["obs"]
    for key in ("agentview_rgb", "agentview_rgb_jpeg"):
        if key in obs:
            return len(obs[key])
    raise KeyError("Neither agentview_rgb nor agentview_rgb_jpeg exists")


def decode_frame(obs: h5py.Group, raw_key: str, jpeg_key: str, index: int) -> np.ndarray:
    if raw_key in obs:
        frame = np.asarray(obs[raw_key][index], dtype=np.uint8)
    elif jpeg_key in obs:
        encoded = np.asarray(obs[jpeg_key][index], dtype=np.uint8)
        with Image.open(io.BytesIO(encoded.tobytes())) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        raise KeyError(f"Neither {raw_key} nor {jpeg_key} exists")
    return np.ascontiguousarray(frame)


def legal_start_bounds(episode_len: int, chunk_size: int, phase: int) -> tuple[int, int]:
    """Return inclusive integer bounds for the requested half-open quartile."""
    lower = math.ceil(phase * episode_len / 4)
    phase_upper_exclusive = math.ceil((phase + 1) * episode_len / 4)
    upper = min(phase_upper_exclusive - 1, episode_len - chunk_size)
    return lower, upper


def eligible_for_all_phases(demo: h5py.Group, chunk_size: int) -> bool:
    episode_len = len(demo["actions"])
    if frame_count(demo) != episode_len or len(demo["robot_states"]) != episode_len:
        return False
    return all(
        lower <= upper
        for lower, upper in (legal_start_bounds(episode_len, chunk_size, phase) for phase in range(4))
    )


def write_pickle(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def build_sample(
    *,
    h5: h5py.File,
    source_path: Path,
    data_root: Path,
    demo_key: str,
    phase: int,
    start: int,
    chunk_size: int,
    sample_index: int,
) -> dict[str, Any]:
    demo = h5[f"data/{demo_key}"]
    obs = demo["obs"]
    actions = np.asarray(demo["actions"], dtype=np.float32)
    proprio = np.asarray(demo["robot_states"], dtype=np.float32)
    episode_len = len(actions)
    end = start + chunk_size
    future_index = min(end, episode_len - 1)  # Matches LIBERODataset future-frame handling.
    lower, upper = legal_start_bounds(episode_len, chunk_size, phase)

    current_observation = {
        "primary_image": decode_frame(obs, "agentview_rgb", "agentview_rgb_jpeg", start),
        "wrist_image": decode_frame(obs, "eye_in_hand_rgb", "eye_in_hand_rgb_jpeg", start),
        "proprio": np.ascontiguousarray(proprio[start]),
    }
    future_observation = {
        "primary_image": decode_frame(obs, "agentview_rgb", "agentview_rgb_jpeg", future_index),
        "wrist_image": decode_frame(obs, "eye_in_hand_rgb", "eye_in_hand_rgb_jpeg", future_index),
        "proprio": np.ascontiguousarray(proprio[future_index]),
    }

    return {
        "format": FORMAT_VERSION,
        "observation": current_observation,
        "task_label": task_label(source_path),
        "action_chunk": np.ascontiguousarray(actions[start:end]),
        "future_observation": future_observation,
        "metadata": {
            "sample_index": sample_index,
            "suite": suite_name(source_path),
            "task_file": source_path.name,
            "source_file": str(source_path.relative_to(data_root)),
            "demo_id": demo_key,
            "episode_length": episode_len,
            "phase": PHASE_NAMES[phase],
            "phase_index": phase,
            "phase_fraction": [phase / 4, (phase + 1) / 4],
            "legal_start_min": lower,
            "legal_start_max": upper,
            "window_start": start,
            "window_end_exclusive": end,
            "future_observation_index": future_index,
            "chunk_size": chunk_size,
        },
    }


def clean_generated_outputs(output_dir: Path) -> None:
    for path in (
        output_dir / "samples",
        output_dir / "libero_calibration_320.pkl",
        output_dir / "manifest.csv",
        output_dir / "manifest.jsonl",
        output_dir / "summary.json",
        output_dir / "dataset_statistics.json",
        output_dir / "dataset_statistics_post_norm.json",
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.demos_per_task != 8:
        raise ValueError("This four-phase design requires exactly 8 demos/task (2 per phase).")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    task_files = sorted(data_dir.glob("*_regen/*.hdf5"))
    if len(task_files) != args.expected_tasks:
        raise RuntimeError(
            f"Expected {args.expected_tasks} task HDF5 files under {data_dir}, found {len(task_files)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_exists = (output_dir / "summary.json").exists() or (output_dir / "samples").exists()
    if generated_exists and not args.overwrite:
        raise FileExistsError(
            f"Generated outputs already exist under {output_dir}; pass --overwrite to replace them"
        )
    if args.overwrite:
        clean_generated_outputs(output_dir)

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for source_path in task_files:
        suite = suite_name(source_path)
        task = task_label(source_path)
        rng = random.Random(task_seed(args.seed, suite, task))
        with h5py.File(source_path, "r") as h5:
            demo_keys = sorted_demo_keys(h5["data"])
            eligible_keys = [
                key for key in demo_keys if eligible_for_all_phases(h5[f"data/{key}"], args.chunk_size)
            ]
            if len(eligible_keys) < args.demos_per_task:
                raise RuntimeError(
                    f"{source_path} has only {len(eligible_keys)} eligible demos; need {args.demos_per_task}"
                )
            selected_keys = rng.sample(eligible_keys, args.demos_per_task)

            for local_index, demo_key in enumerate(selected_keys):
                phase = local_index // 2
                episode_len = len(h5[f"data/{demo_key}/actions"])
                lower, upper = legal_start_bounds(episode_len, args.chunk_size, phase)
                start = rng.randint(lower, upper)
                sample_index = len(records)
                record = build_sample(
                    h5=h5,
                    source_path=source_path,
                    data_root=data_dir,
                    demo_key=demo_key,
                    phase=phase,
                    start=start,
                    chunk_size=args.chunk_size,
                    sample_index=sample_index,
                )

                task_slug = source_path.stem.removesuffix("_demo")
                relative_sample_path = (
                    Path("samples")
                    / suite
                    / task_slug
                    / f"{sample_index:03d}_{PHASE_NAMES[phase]}_{demo_key}_t{start}.pkl"
                )
                sample_path = output_dir / relative_sample_path
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                write_pickle(sample_path, record)
                records.append(record)
                manifest_rows.append(
                    {
                        **record["metadata"],
                        "task_label": task,
                        "sample_file": str(relative_sample_path),
                    }
                )

    aggregate_path = output_dir / "libero_calibration_320.pkl"
    write_pickle(aggregate_path, records)

    fieldnames = list(manifest_rows[0].keys())
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for stats_name in ("dataset_statistics.json", "dataset_statistics_post_norm.json"):
        source = data_dir / stats_name
        if source.exists():
            shutil.copy2(source, output_dir / stats_name)

    suite_counts = Counter(row["suite"] for row in manifest_rows)
    phase_counts = Counter(row["phase"] for row in manifest_rows)
    task_counts = Counter((row["suite"], row["task_label"]) for row in manifest_rows)
    demo_pairs = {(row["source_file"], row["demo_id"]) for row in manifest_rows}
    summary = {
        "format": FORMAT_VERSION,
        "source_data": str(data_dir),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "num_tasks": len(task_counts),
        "num_windows": len(records),
        "num_distinct_demonstrations": len(demo_pairs),
        "demos_per_task": args.demos_per_task,
        "windows_per_demo": 1,
        "windows_per_task": dict(sorted(Counter(task_counts.values()).items())),
        "suite_counts": dict(sorted(suite_counts.items())),
        "phase_counts": dict((name, phase_counts[name]) for name in PHASE_NAMES),
        "aggregate_file": aggregate_path.name,
        "individual_samples_dir": "samples",
        "manifest_csv": "manifest.csv",
        "manifest_jsonl": "manifest.jsonl",
    }
    if summary["num_windows"] != args.expected_tasks * args.demos_per_task:
        raise AssertionError(f"Unexpected window count: {summary['num_windows']}")
    if summary["num_distinct_demonstrations"] != summary["num_windows"]:
        raise AssertionError("A demonstration was sampled more than once")
    if set(task_counts.values()) != {args.demos_per_task}:
        raise AssertionError(f"Per-task counts are not uniform: {Counter(task_counts.values())}")
    if any(phase_counts[name] != args.expected_tasks * 2 for name in PHASE_NAMES):
        raise AssertionError(f"Phase counts are not balanced: {phase_counts}")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
