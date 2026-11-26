"""Initialize LIBERO asset paths without its first-import interactive prompt."""

import importlib.util
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


def prepare_libero():
    import yaml

    spec = importlib.util.find_spec("libero")
    if spec is None:
        # Upstream setup.py uses find_packages() for a namespace package;
        # modern editable installs can omit its import-path mapping.
        distribution = importlib.metadata.distribution("libero")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        source = urlsplit(direct.get("url", ""))
        source_root = Path(unquote(source.path))
        if source.scheme == "file" and (source_root / "libero/libero").is_dir():
            sys.path.insert(0, str(source_root))
            spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("LIBERO is missing; install requirements.txt first")
    configured = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    if (configured / "config.yaml").is_file():
        return
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    for name in ("bddl_files", "init_files", "assets"):
        if not (root / name).is_dir():
            raise FileNotFoundError(
                f"Missing LIBERO assets: {root / name}. Use the editable installation in requirements.txt."
            )
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "steerquant" / "libero"
    destination = configured if "LIBERO_CONFIG_PATH" in os.environ else cache
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "assets": str(root / "assets"),
        "datasets": str(root.parent / "datasets"),
    }
    (destination / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(destination)
