import pickle

import h5py
import numpy as np
import pytest

from steerquant.prepare_robocasa_calibration import CAMERAS, RemoteH5, collect_task, sha


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_text_cache_label_check_skips_tensor_payloads(tmp_path, kind):
    from steerquant.prepare_robocasa_calibration import text_cache_labels
    import torch

    value = np.zeros((1, 512, 1024), dtype=np.float16)
    if kind == "torch":
        value = torch.from_numpy(value)
    path = tmp_path / "t5.pkl"
    path.write_bytes(pickle.dumps({"task one": value, "task two": value}))
    assert text_cache_labels(path) == {"task one", "task two"}


def source_fixture(root):
    source = root / "success_only/kitchen_coffee/CoffeePressButton/train/demo.hdf5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as handle:
        for i in range(8):
            demo = handle.create_group(f"data/demo_{i}")
            demo.attrs["task_description"] = "press the coffee button"
            demo.create_dataset("actions", (160, 12), dtype="f4", fillvalue=i / 10)
            demo.create_dataset("robot_states", (160, 9), dtype="f4", fillvalue=i)
            demo.create_dataset("rewards", (160,), dtype="f4", fillvalue=1)
            for camera in CAMERAS.values():
                demo.create_dataset(
                    f"obs/{camera}", (160, 224, 224, 3), dtype="u1", chunks=(1, 224, 224, 3), fillvalue=i
                )
    item = dict(
        path=str(source.relative_to(root)).replace("\\", "/"),
        size=source.stat().st_size,
        lfs=dict(sha256=sha(source)),
    )
    inventory = dict(repo="test/fixture", revision="a" * 40, files=[item])
    return inventory, item


def test_local_sampling_preserves_three_cameras_actions_and_balanced_windows(tmp_path):
    inventory, item = source_fixture(tmp_path / "input")
    results = []
    for name in ("first", "second"):
        out = tmp_path / name
        out.mkdir()
        report = collect_task((inventory, item, 195, 32, out, tmp_path / "input"))
        rows = report["samples"]
        assert len(rows) == len({r["demo_id"] for r in rows}) == 8
        assert [r["phase_index"] for r in rows] == [0, 0, 1, 1, 2, 2, 3, 3]
        payloads = [(out / r["sample_file"]).read_bytes() for r in rows]
        results.append(payloads)
        for row, payload in zip(rows, payloads):
            record = pickle.loads(payload)
            assert record["action_chunk"].shape == (32, 7)
            assert record["observation"]["proprio"].shape == (9,)
            assert record["metadata"]["source_full_file_hash_verified"]
            assert record["metadata"]["window_start"] + 32 <= record["metadata"]["episode_length"]
            for camera in CAMERAS:
                assert record["observation"][camera].shape == (224, 224, 3)
                assert record["observation"][camera].dtype == np.uint8
        assert collect_task((inventory, item, 195, 32, out, tmp_path / "input")) == report
        with pytest.raises(AssertionError):
            collect_task((inventory, item, 196, 32, out, tmp_path / "input"))
    assert results[0] == results[1]


def test_local_source_checksum_mismatch_is_rejected(tmp_path):
    inventory, item = source_fixture(tmp_path / "input")
    item["lfs"]["sha256"] = "0" * 64
    out = tmp_path / "output"
    out.mkdir()
    with pytest.raises(ValueError, match="SHA256"):
        collect_task((inventory, item, 195, 32, out, tmp_path / "input"))


def test_range_reader_reuses_and_validates_cached_blocks(tmp_path, monkeypatch):
    content = bytes(range(256)) * 700
    inventory = dict(repo="test/fixture", revision="a" * 40)
    item = dict(path="success_only/category/task/demo.hdf5", size=len(content))
    calls = []

    def fetch(self, start, end):
        calls.append((start, end))
        return content[start : end + 1]

    monkeypatch.setattr(RemoteH5, "fetch", fetch)
    with RemoteH5(inventory, item, tmp_path) as reader:
        reader.seek(65000)
        assert reader.read(70000) == content[65000:135000]
        reader.seek(-30, 2)
        assert reader.read(100) == content[-30:]
    before = len(calls)
    with RemoteH5(inventory, item, tmp_path) as reader:
        reader.seek(65000)
        assert reader.read(70000) == content[65000:135000]
    assert len(calls) == before
    first = tmp_path / "source_ranges/task/00000000.bin"
    first.write_bytes(b"corrupt")
    with RemoteH5(inventory, item, tmp_path) as reader:
        with pytest.raises(AssertionError):
            reader.read(10)
