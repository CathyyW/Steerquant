"""Inspect a released packed checkpoint without constructing the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .deployment import validate_payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_sha256(path: Path) -> str | None:
    manifest = path.parent / "SHA256SUMS"
    if not manifest.is_file():
        return None
    matches = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split(maxsplit=1)
        if name.lstrip("*") == path.name:
            matches.append(checksum)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {path.name} entry in {manifest}.")
    return matches[0]


def inspect_checkpoint(path: str | Path, *, verify_checksum: bool = True) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    expected = expected_sha256(resolved) if verify_checksum else None
    actual = sha256_file(resolved) if expected else None
    if expected and actual != expected:
        raise ValueError(f"SHA256 mismatch for {resolved}: expected {expected}, got {actual}")

    payload = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
    validate_payload(payload)
    model = payload["model_config"]
    robot = model.get("real_robot", {})
    rotations = sorted({site["rotation"] for site in payload["sites"]})
    summary = {
        "path": str(resolved),
        "sha256": actual,
        "format": payload["format"],
        "weight_bits": payload["weight_bits"],
        "activation_bits": payload["activation_bits"],
        "num_denoise_calls": payload["num_calls"],
        "quantized_sites": len(payload["sites"]),
        "rotations": rotations,
        "task": robot.get("task"),
        "prompt": robot.get("prompt"),
        "action_horizon": model.get("model", {}).get("action_horizon"),
        "num_video_frames": model.get("model", {}).get("num_video_frames"),
        "source_checkpoint_sha256": robot.get("source_checkpoint_sha256"),
    }
    del payload
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to deployment.pt")
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip SHA256SUMS verification when a sibling manifest exists.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(
        inspect_checkpoint(args.checkpoint, verify_checksum=not args.no_checksum),
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
