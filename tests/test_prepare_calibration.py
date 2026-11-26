import pickle
import sys

import h5py
import numpy as np

from steerquant.prepare_calibration import main


def test_sampling_is_reproducible_and_balanced(tmp_path, monkeypatch):
    source = tmp_path / "data" / "libero_spatial_regen"
    source.mkdir(parents=True)
    with h5py.File(source / "pick_up_the_black_bowl_demo.hdf5", "w") as handle:
        for index in range(10):
            demo = handle.create_group(f"data/demo_{index}")
            demo["actions"] = np.zeros((80, 7), np.float32)
            demo["robot_states"] = np.full((80, 9), index, np.float32)
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                demo[f"obs/{key}"] = np.full((80, 8, 8, 3), index, np.uint8)
    outputs = []
    for name in ("first", "second"):
        destination = tmp_path / name
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prepare",
                "--data-dir",
                str(source.parent),
                "--output-dir",
                str(destination),
                "--expected-tasks",
                "1",
            ],
        )
        main()
        outputs.append((destination / "libero_calibration_320.pkl").read_bytes())
    assert outputs[0] == outputs[1]
    records = pickle.loads(outputs[0])
    assert len(records) == len({r["metadata"]["demo_id"] for r in records}) == 8
    assert sorted(r["metadata"]["phase_index"] for r in records) == [0, 0, 1, 1, 2, 2, 3, 3]
    for record in records:
        assert record["action_chunk"].shape == (16, 7)
        assert record["observation"]["proprio"].shape == (9,)
        metadata = record["metadata"]
        assert metadata["window_start"] + 16 <= metadata["episode_length"]
