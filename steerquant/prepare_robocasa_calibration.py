#!/usr/bin/env python3
"""Pinned, task-balanced training-only RoboCasa calibration via HTTP byte ranges.
No simulator/model evaluation. Raw images/proprio; action[:7], H=32.
"""

import argparse, collections, concurrent.futures, csv, hashlib, io, json, math
import pickle, random, time
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import quote
import h5py
import numpy as np
import requests
from PIL import Image

DEFAULT_INVENTORY = Path(__file__).with_name("data") / "robocasa_sources.json"
PHASES = ("early", "early_middle", "late_middle", "late")
CAMERAS = {
    "primary_image": "robot0_agentview_left_rgb",
    "secondary_image": "robot0_agentview_right_rgb",
    "wrist_image": "robot0_eye_in_hand_rgb",
}
FORMAT = "cosmos_policy.robocasa.calibration_window.v1"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p, d):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(p)


def write_pickle(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


class RemoteH5(io.RawIOBase):
    """64-KiB cached, exact HTTP ranges; no full-sized sparse local file.
    Each block is locally SHA256 verified on reuse. Whole source LFS hashes
    are provenance only: a partial download cannot verify the full-file hash.
    """

    BS = 65536

    def __init__(self, inventory, item, output_dir):
        super().__init__()
        self.pos = 0
        self.size = item["size"]
        self.item = item
        self.source = f"https://huggingface.co/datasets/{inventory['repo']}/resolve/{inventory['revision']}/{quote(item['path'], safe='/')}"
        self.cache = Path(output_dir) / "source_ranges" / item["path"].split("/")[2]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.meta = self.cache / "ranges.json"
        self.state = (
            json.loads(self.meta.read_text())
            if self.meta.exists()
            else {
                "source_file": item["path"],
                "revision": inventory["revision"],
                "size": self.size,
                "block_size": self.BS,
                "blocks": {},
            }
        )
        assert (
            self.state["revision"] == inventory["revision"]
            and self.state["source_file"] == item["path"]
            and self.state["size"] == self.size
        )
        self.resolve_session = requests.Session()
        self.direct = requests.Session()
        self.direct.trust_env = True
        self.url = None
        self.new_bytes = 0
        self.requests = 0
        self.memory = collections.OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        p = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if p < 0:
            raise ValueError("negative seek")
        self.pos = p
        return p

    def resolve(self):
        with self.resolve_session.get(
            self.source + f"?download=true&resolve={time.time_ns()}",
            headers={"Range": "bytes=0-0"},
            timeout=(15, 30),
            stream=True,
        ) as r:
            r.raise_for_status()
            assert r.status_code == 206 and r.headers.get("Content-Range") == f"bytes 0-0/{self.size}"
            self.url = r.url

    def fetch(self, a, b):
        for attempt in range(6):
            try:
                if not self.url:
                    self.resolve()
                with self.direct.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=(15, 30)) as r:
                    r.raise_for_status()
                    assert (
                        r.status_code == 206
                        and r.headers.get("Content-Range") == f"bytes {a}-{b}/{self.size}"
                    )
                    data = r.content
                    assert len(data) == b - a + 1
                self.requests += 1
                self.new_bytes += len(data)
                return data
            except Exception as e:
                print(self.cache.name, "range retry", attempt + 1, type(e).__name__, flush=True)
                if attempt == 5:
                    raise RuntimeError(f"Range failed for {self.item['path']} {a}:{b}") from None
                self.url = None
                time.sleep(1 + attempt)

    def block(self, i):
        if i in self.memory:
            self.memory.move_to_end(i)
            return self.memory[i]
        p = self.cache / f"{i:08d}.bin"
        entry = self.state["blocks"].get(str(i))
        if entry and p.exists():
            data = p.read_bytes()
            assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], str(p)
        else:
            a = i * self.BS
            b = min(a + self.BS, self.size) - 1
            data = self.fetch(a, b)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
            self.state["blocks"][str(i)] = {
                "offset": a,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            atomic_json(self.meta, self.state)
        self.memory[i] = data
        if len(self.memory) > 256:
            self.memory.popitem(last=False)
        return data

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, max(0, self.size - self.pos))
        if not n:
            return b""
        start = self.pos
        end = start + n
        first = start // self.BS
        last = (end - 1) // self.BS
        # Coalesce consecutive missing blocks into one exact request.
        i = first
        while i <= last:
            if str(i) in self.state["blocks"]:
                i += 1
                continue
            j = i
            while j + 1 <= last and str(j + 1) not in self.state["blocks"] and j - i < 127:
                j += 1
            a = i * self.BS
            b = min((j + 1) * self.BS, self.size) - 1
            data = self.fetch(a, b)
            for k in range(i, j + 1):
                piece = data[(k - i) * self.BS : (k - i + 1) * self.BS]
                p = self.cache / f"{k:08d}.bin"
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(piece)
                tmp.replace(p)
                self.state["blocks"][str(k)] = {
                    "offset": k * self.BS,
                    "bytes": len(piece),
                    "sha256": hashlib.sha256(piece).hexdigest(),
                }
            atomic_json(self.meta, self.state)
            i = j + 1
        result = b"".join(self.block(i) for i in range(first, last + 1))[
            start - first * self.BS : start - first * self.BS + n
        ]
        assert len(result) == n
        self.pos = end
        return result

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self):
        if not self.closed:
            self.direct.close()
            self.resolve_session.close()
        super().close()


def bounds(T, H, q):
    return math.ceil(q * T / 4), min(math.ceil((q + 1) * T / 4) - 1, T - H)


def task_seed(root, category, task):
    return int.from_bytes(hashlib.sha256(f"{root}\0{category}\0{task}".encode()).digest()[:8], "big")


def frame(group, key, t):
    if key in group:
        raw = group[key][t]
        assert raw.dtype == np.uint8
        out = np.asarray(raw)
    else:
        encoded = np.asarray(group[key + "_jpeg"][t], dtype=np.uint8)
        with Image.open(io.BytesIO(encoded.tobytes())) as im:
            out = np.asarray(im.convert("RGB"))
    assert out.shape == (224, 224, 3)
    return np.ascontiguousarray(out)


def collect_task(job):
    inventory, item, seed, H, output_dir, data_dir = job
    OUT = Path(output_dir)
    category, task = item["path"].split("/")[1:3]
    report_path = OUT / "tasks" / f"{task}.json"
    report_path.parent.mkdir(exist_ok=True)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert (
            report["revision"] == inventory["revision"]
            and report["seed"] == seed
            and report["chunk_size"] == H
        )
        for row in report["samples"]:
            assert sha(OUT / row["sample_file"]) == row["sample_sha256"]
        print("REUSE", task, flush=True)
        return report
    print("START", task, flush=True)
    started = time.monotonic()
    remote = None
    if data_dir is None:
        remote = RemoteH5(inventory, item, OUT)
        source = remote
    else:
        source = Path(data_dir) / item["path"]
        if not source.is_file() or source.stat().st_size != item["size"]:
            raise ValueError(f"Source size differs from pinned inventory: {source}")
        expected = item.get("lfs", {}).get("sha256")
        if expected and sha(source) != expected:
            raise ValueError(f"Source SHA256 differs from pinned inventory: {source}")
    excluded = []
    samples = []
    demo_inventory = []
    eligible = []
    with remote if remote is not None else nullcontext(), h5py.File(source, "r") as h:
        keys = sorted(h["data"], key=lambda k: int(k.rsplit("_", 1)[1]))
        for key in keys:
            d = h["data"][key]
            T = len(d["actions"])
            reasons = []
            if any(lo > hi for lo, hi in (bounds(T, H, q) for q in range(4))):
                reasons.append("no_legal_start_in_all_four_quartiles")
            if d["actions"].ndim != 2 or d["actions"].shape[1] < 7:
                reasons.append("invalid_action_shape")
            if d["robot_states"].shape != (T, 9):
                reasons.append("invalid_proprio_shape_or_length")
            camera_shapes = {}
            for cam in CAMERAS.values():
                name = cam if cam in d["obs"] else cam + "_jpeg"
                if name not in d["obs"]:
                    reasons.append("missing_" + cam)
                    continue
                arr = d["obs"][name]
                camera_shapes[name] = list(arr.shape)
                if len(arr) != T:
                    reasons.append("camera_length_mismatch_" + cam)
                if name == cam and (arr.shape[1:] != (224, 224, 3) or arr.dtype != np.uint8):
                    reasons.append("invalid_camera_" + cam)
            if "task_description" not in d.attrs:
                reasons.append("missing_task_description")
            row = {
                "demo_id": key,
                "episode_length": T,
                "actions_shape": list(d["actions"].shape),
                "camera_shapes": camera_shapes,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
            demo_inventory.append(row)
            if reasons:
                excluded.append(row)
            else:
                eligible.append(key)
        assert len(eligible) >= 8, (task, len(eligible))
        print("ELIGIBLE", task, len(eligible), "/", len(keys), flush=True)
        derived = task_seed(seed, category, task)
        rng = random.Random(derived)
        chosen = rng.sample(eligible, 8)
        for i, key in enumerate(chosen):
            q = i // 2
            d = h["data"][key]
            T = len(d["actions"])
            lo, hi = bounds(T, H, q)
            t = rng.randint(lo, hi)
            future = min(t + H, T - 1)
            label = d.attrs["task_description"]
            label = label.decode() if isinstance(label, bytes) else str(label)

            def observation(index):
                obs = {dest: frame(d["obs"], src, index) for dest, src in CAMERAS.items()}
                obs["proprio"] = np.ascontiguousarray(d["robot_states"][index], dtype=np.float32)
                assert obs["proprio"].shape == (9,) and np.isfinite(obs["proprio"]).all()
                return obs

            actions = np.ascontiguousarray(d["actions"][t : t + H, :7], dtype=np.float32)
            assert actions.shape == (H, 7) and np.isfinite(actions).all()
            # Check source success signal rather than inventing an episode label.
            terminal_reward = float(d["rewards"][-1]) if "rewards" in d else None
            terminal_done = int(d["dones"][-1]) if "dones" in d else None
            assert terminal_reward is None or terminal_reward == 1, (task, key, terminal_reward)
            rel = f"samples/{task}/{i:02d}_{PHASES[q]}_{key}_t{t}.pkl"
            metadata = {
                "task": task,
                "category": category,
                "source_repo": inventory["repo"],
                "source_revision": inventory["revision"],
                "source_split": "success_only",
                "source_file": item["path"],
                "source_file_bytes": item["size"],
                "source_lfs_sha256": item["lfs"]["sha256"],
                "source_full_file_hash_verified": remote is None,
                "demo_id": key,
                "episode_length": T,
                "phase": PHASES[q],
                "phase_index": q,
                "legal_start_min": lo,
                "legal_start_max": hi,
                "window_start": t,
                "window_end_exclusive": t + H,
                "future_observation_index": future,
                "chunk_size": H,
                "sampling_seed": seed,
                "task_sampling_seed": derived,
                "evaluation_seed": None,
                "task_sample_index": i,
                "source_action_dim": d["actions"].shape[1],
                "action_columns_used": list(range(7)),
                "terminal_reward": terminal_reward,
                "terminal_done": terminal_done,
                "image_transform": "none; source images already flipped during official regeneration",
                "normalization": "raw; apply checkpoint statistics during inference",
            }
            record = {
                "format": FORMAT,
                "observation": observation(t),
                "task_label": label,
                "action_chunk": actions,
                "future_observation": observation(future),
                "metadata": metadata,
            }
            write_pickle(OUT / rel, record)
            samples.append(
                {**metadata, "task_label": label, "sample_file": rel, "sample_sha256": sha(OUT / rel)}
            )
            print("SAMPLE", task, i + 1, "/8", key, "t", t, flush=True)
    report = {
        "task": task,
        "revision": inventory["revision"],
        "seed": seed,
        "chunk_size": H,
        "source_file": item["path"],
        "total_demos": len(keys),
        "eligible_demos": len(eligible),
        "excluded_demos": excluded,
        "demo_inventory": demo_inventory,
        "samples": samples,
        "new_download_bytes": remote.new_bytes if remote else 0,
        "range_requests": remote.requests if remote else 0,
        "seconds": time.monotonic() - started,
    }
    atomic_json(report_path, report)
    print("DONE", task, round(report["seconds"], 1), "s", flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional local dataset root containing success_only/; otherwise fetch exact HTTP ranges",
    )
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--task", help="Prepare one task; rerun without this option to aggregate all 24")
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument(
        "--validate-only", action="store_true", help="Verify existing records without downloading or writing"
    )
    parser.add_argument(
        "--t5-embeddings", type=Path, help="Optionally verify task labels against the model text cache"
    )
    return parser


class _DiscardedArray:
    def __setstate__(self, state):
        pass


def text_cache_labels(path):
    """Read cache keys without materializing hundreds of MiB of tensor values."""

    class LabelsUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("torch") and (name.startswith("_rebuild") or name == "_load_from_bytes"):
                return lambda *args, **kwargs: None
            if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name == "_reconstruct":
                return lambda *args: _DiscardedArray()
            if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
                return lambda *args: _DiscardedArray()
            if module == "numpy" and name == "ndarray":
                return _DiscardedArray
            if module == "numpy" and name == "dtype":
                return np.dtype
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"Unsupported text-cache value type: {module}.{name}")

    with Path(path).open("rb") as handle:
        cache = LabelsUnpickler(handle).load()
    if not isinstance(cache, dict) or not all(isinstance(key, str) for key in cache):
        raise ValueError("Expected a text embedding dictionary with string labels")
    return set(cache)


def validate_dataset(root, *, inventory=None, t5_embeddings=None):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    aggregate = root / "robocasa_calibration_192.pkl"
    if sha(aggregate) != summary["aggregate_sha256"]:
        raise ValueError("Aggregate SHA256 mismatch")
    with aggregate.open("rb") as handle:
        records = pickle.load(handle)
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    if len(records) != 192 or len(rows) != 192:
        raise ValueError("Expected 192 calibration samples")
    counts = collections.Counter()
    identities = set()
    labels = set()
    sources = {item["path"]: item for item in inventory["files"]} if inventory else None
    for record, row in zip(records, rows):
        meta = record["metadata"]
        task = meta["task"]
        phase = meta["phase_index"]
        if record["format"] != FORMAT or meta["source_split"] != "success_only":
            raise ValueError("Unexpected sample format or split")
        if inventory and (
            meta["source_repo"] != inventory["repo"]
            or meta["source_revision"] != inventory["revision"]
            or meta["source_file"] not in sources
        ):
            raise ValueError("Sample does not match pinned source inventory")
        if any(row.get(k) != v for k, v in meta.items()) or row["task_label"] != record["task_label"]:
            raise ValueError("Manifest/sample metadata mismatch")
        sample_path = (root / row["sample_file"]).resolve()
        if not sample_path.is_relative_to(root.resolve()) or sha(sample_path) != row["sample_sha256"]:
            raise ValueError("Sample path or SHA256 mismatch")
        with sample_path.open("rb") as handle:
            individual = pickle.load(handle)
        if individual["metadata"] != meta or individual["task_label"] != record["task_label"]:
            raise ValueError("Aggregate/per-sample metadata mismatch")
        lo, hi = bounds(meta["episode_length"], 32, phase)
        if phase not in range(4) or not lo <= meta["window_start"] <= hi or meta["chunk_size"] != 32:
            raise ValueError("Invalid action window")
        if meta["window_end_exclusive"] != meta["window_start"] + 32:
            raise ValueError("Action-window end mismatch")
        key = (meta["source_file"], meta["demo_id"])
        if key in identities:
            raise ValueError("Repeated demonstration")
        identities.add(key)
        counts[task, phase] += 1
        labels.add(record["task_label"])
        actions = np.asarray(record["action_chunk"])
        if actions.shape != (32, 7) or actions.dtype != np.float32 or not np.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        if not np.array_equal(actions, individual["action_chunk"]):
            raise ValueError("Aggregate/per-sample actions mismatch")
        for field in ("observation", "future_observation"):
            obs = record[field]
            for camera in CAMERAS:
                image = np.asarray(obs[camera])
                if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid camera: {camera}")
                if not np.array_equal(image, individual[field][camera]):
                    raise ValueError("Aggregate/per-sample camera mismatch")
            proprio = np.asarray(obs["proprio"])
            if proprio.shape != (9,) or proprio.dtype != np.float32 or not np.isfinite(proprio).all():
                raise ValueError("Invalid proprioception")
            if not np.array_equal(proprio, individual[field]["proprio"]):
                raise ValueError("Aggregate/per-sample proprioception mismatch")
    from .integrations.robocasa_protocol import TASKS

    if set(task for task, phase in counts) != set(TASKS) or len(counts) != 96 or set(counts.values()) != {2}:
        raise ValueError("Expected all 24 tasks with two samples in each of four phases")
    sample_count = len(records)
    del records, individual
    if t5_embeddings:
        if not labels.issubset(text_cache_labels(t5_embeddings)):
            raise ValueError("Task labels missing from the model T5 cache")
    return dict(
        passed=True,
        samples=sample_count,
        tasks=24,
        distinct_demonstrations=len(identities),
        aggregate_sha256=summary["aggregate_sha256"],
        all_sample_hashes_verified=True,
        text_labels_verified=bool(t5_embeddings),
    )


def main():
    args = build_parser().parse_args()
    OUT = args.output_dir.expanduser().resolve()
    inventory = json.loads(args.source_inventory.read_text())
    if args.validate_only:
        print(
            json.dumps(validate_dataset(OUT, inventory=inventory, t5_embeddings=args.t5_embeddings), indent=2)
        )
        return
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    files = sorted([f for f in inventory["files"] if f["path"].endswith(".hdf5")], key=lambda f: f["path"])
    from .integrations.robocasa_protocol import TASKS

    if len(files) != 24 or {f["path"].split("/")[2] for f in files} != set(TASKS):
        raise ValueError("Source inventory must contain all 24 RoboCasa tasks")
    if args.task:
        files = [f for f in files if f["path"].split("/")[2] == args.task]
        if len(files) != 1:
            raise ValueError("Unknown RoboCasa task")
    OUT.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT / "source_inventory.json"
    if inventory_path.exists() and json.loads(inventory_path.read_text()) != inventory:
        raise ValueError("Output directory belongs to another source inventory")
    atomic_json(inventory_path, inventory)
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir else None
    jobs = [(inventory, f, args.seed, 32, OUT, data_dir) for f in files]
    # Single worker also supports deterministic offline tests without process spawning.
    if args.workers == 1:
        reports = list(map(collect_task, jobs))
    else:
        with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
            reports = list(pool.map(collect_task, jobs))
    if args.task:
        return
    rows = [row for r in reports for row in r["samples"]]
    records = []
    assert len(rows) == 192
    assert len({(r["source_file"], r["demo_id"]) for r in rows}) == 192
    for i, row in enumerate(rows):
        with (OUT / row["sample_file"]).open("rb") as f:
            records.append(pickle.load(f))
        row["sample_index"] = i
    destination = OUT / "robocasa_calibration_192.pkl"
    write_pickle(destination, records)
    with (OUT / "manifest.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with (OUT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": FORMAT,
        "source_repo": inventory["repo"],
        "source_revision": inventory["revision"],
        "source_split": "success_only",
        "tasks": 24,
        "samples": len(records),
        "distinct_demonstrations": 192,
        "samples_per_task": 8,
        "samples_per_task_per_quartile": 2,
        "sampling_seed": args.seed,
        "seed_meaning": "sampling only; no evaluation performed",
        "action_chunk_size": 32,
        "action_shape": [32, 7],
        "proprio_shape": [9],
        "camera_shape": [224, 224, 3],
        "cameras": list(CAMERAS),
        "aggregate_file": destination.name,
        "aggregate_sha256": sha(destination),
        "normalized": False,
        "source_full_files_downloaded": bool(data_dir),
        "source_full_file_sha256_verified": bool(data_dir),
        "verification": (
            "Full local files and samples SHA256 verified"
            if data_dir
            else "Pinned HTTPS exact byte ranges and sizes; local range/sample SHA256; source full-file SHA256 recorded only"
        ),
        "policy_forward_run": False,
        "activation_statistics_collected": False,
        "eligible_demo_counts": {r["task"]: r["eligible_demos"] for r in reports},
        "excluded_demo_counts": {r["task"]: len(r["excluded_demos"]) for r in reports},
    }
    atomic_json(OUT / "summary.json", summary)
    print(json.dumps(validate_dataset(OUT, inventory=inventory, t5_embeddings=args.t5_embeddings), indent=2))


if __name__ == "__main__":
    main()
