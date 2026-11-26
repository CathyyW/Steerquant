from __future__ import annotations
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from steerquant.build import cutlass_root


@lru_cache(maxsize=1)
def load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("The CUTLASS W4A8 backend requires CUDA.")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    if shutil.which("ninja") is None:
        environment_bin = str(Path(sys.executable).parent)
        if (Path(environment_bin) / "ninja").is_file():
            os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    source_root = Path(__file__).resolve().parent / "csrc"
    headers_root = cutlass_root()
    return load(
        name="steerquant_w4a8_v1",
        sources=[str(source_root / "binding.cpp"), str(source_root / "w4a8.cu")],
        extra_include_paths=[str(headers_root / "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo"],
        with_cuda=True,
        verbose=os.environ.get("COSMOS_W4A8_BUILD_VERBOSE", "0") == "1",
    )
