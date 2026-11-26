"""Fetch pinned CUTLASS headers and build both native CUDA extensions."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
from filelock import FileLock

CUTLASS_TAG = "v3.8.0"
CUTLASS_REVISION = "afa1772203677c5118fcd82537a9c8fefbcc7008"


def cutlass_root():
    if os.environ.get("CUTLASS_PATH"):
        root = Path(os.environ["CUTLASS_PATH"]).expanduser().resolve()
        if not (root / "include/cutlass/cutlass.h").is_file():
            raise FileNotFoundError(f"Invalid CUTLASS_PATH: {root}")
        return root
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "steerquant"
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / f"cutlass-{CUTLASS_TAG}"
    with FileLock(str(root) + ".lock"):
        if not (root / "include/cutlass/cutlass.h").is_file():
            url = f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_REVISION}"
            with tempfile.TemporaryDirectory(dir=cache, prefix="cutlass-download-") as temporary:
                staging = Path(temporary)
                archive = staging / "source.tar.gz"
                with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
                headers = staging / "headers"
                headers.mkdir()
                with tarfile.open(archive, "r:gz") as source:
                    for member in source:
                        parts = Path(member.name).parts[1:]
                        if not member.isfile() or not parts or ".." in parts:
                            continue
                        if parts[0] != "include" and parts != ("LICENSE.txt",):
                            continue
                        destination = headers.joinpath(*parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(member) as data, destination.open("wb") as handle:
                            shutil.copyfileobj(data, handle)
                if not (headers / "include/cutlass/cutlass.h").is_file():
                    raise RuntimeError("CUTLASS source archive did not contain the required headers")
                headers.rename(root)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Build on an NVIDIA SM80+ CUDA GPU")
    if args.verbose:
        os.environ["COSMOS_W4A8_BUILD_VERBOSE"] = "1"
        os.environ["COSMOS_W4A4_BUILD_VERBOSE"] = "1"
    print(f"CUTLASS: {cutlass_root()}", flush=True)
    from .kernels.w4a8.loader import load_extension as load_w4a8
    from .kernels.w4a4.loader import load_extension as load_w4a4

    for bits, loader in ((8, load_w4a8), (4, load_w4a4)):
        extension = loader()
        print(f"W4A{bits}: {extension.__file__}", flush=True)


if __name__ == "__main__":
    main()
