import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fastwam_steerquant.adapters.preprocessing import WAMObservation, preprocess_real_images
from fastwam_steerquant.adapters.real_robot import (
    FASTWAM_REVISION,
    _sync_vae_scale_device,
    public_fastwam_config,
)
from fastwam_steerquant.robot_async_server import (
    audit_observation_sha256,
    synthetic_audit_observation,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "stats/pack/dataset_stats.json": "30561edb2897b9784fbf13f4e42ca9897e141b4ebd60119d047d255758aaa6bf",
    "stats/stack/dataset_stats.json": "68c662dba501b1e110e09896238bfd0ae1d15c226e2c05683958bba915f7e0cb",
    "text_embeds/pack/ad085ca9b81861e511916a552aef91b865cb6b0b0c6e2c7c3d7db8629eab5ba3.t5_len128.wan22ti2v5b.pt":
        "927573ff154e4303f06a3a719167c198b0b4ed5ea1bd72b23768217fd8381960",
    "text_embeds/stack/072dbf0aee70fe628914d55762f3732ffb939ec8e9af90d5f0b49117f4680b0e.t5_len128.wan22ti2v5b.pt":
        "a83dfdf3a215e8314206bbf5080565617c33e23622ad05a18dd3227afa5fddf6",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_bundled_asset_identities_and_shapes():
    for relative, expected in ASSETS.items():
        assert sha256(ROOT / "assets" / relative) == expected
    for task in ("pack", "stack"):
        config = yaml.safe_load((ROOT / "configs" / f"robot_{task}_w4a8.example.yaml").read_text())
        assert config["wam"]["quantization"]["fastwam_revision"] == FASTWAM_REVISION
        assert "offline" not in config
        cache = torch.load(ROOT / config["wam"]["context_cache"], map_location="cpu",
                           weights_only=True, mmap=True)
        assert cache["context"].shape == (128, 4096)
        assert cache["mask"].shape == (128,)


def test_public_fastwam_target_mapping_is_narrow():
    config = {"_target_": "qi.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor"}
    assert public_fastwam_config(config)["_target_"].startswith("fastwam.datasets.")
    with pytest.raises(ValueError, match="Unsupported private qi target"):
        public_fastwam_config({"_target_": "qi.deployment.private.Policy"})


def test_public_fastwam_vae_scale_is_moved_with_model_precision():
    class VAE:
        model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)
        mean = torch.ones(2, dtype=torch.float32)
        std = torch.full((2,), 2.0, dtype=torch.float32)

    class Model:
        vae = VAE()

    model = Model()
    _sync_vae_scale_device(model)
    assert model.vae.mean.dtype == torch.bfloat16
    assert model.vae.std.dtype == torch.bfloat16
    assert all(value.device == model.vae.model.weight.device for value in model.vae.scale)
    torch.testing.assert_close(model.vae.scale[1].float(), torch.full((2,), 0.5))


def test_three_camera_mosaic_contract():
    shape = (480, 640, 3)
    high = np.zeros(shape, dtype=np.uint8)
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    high[..., 0] = 255
    left[..., 1] = 255
    right[..., 2] = 255
    obs = WAMObservation(high, left, right, np.zeros(14, dtype=np.float32), "task")
    image = preprocess_real_images(obs, torch.device("cpu"), torch.float32)
    assert image.shape == (1, 3, 384, 320)
    torch.testing.assert_close(image[0, :, 64, 160], torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(image[0, :, 320, 80], torch.tensor([-1.0, 1.0, -1.0]))
    torch.testing.assert_close(image[0, :, 320, 240], torch.tensor([-1.0, -1.0, 1.0]))


@pytest.mark.parametrize("task,expected_sha256", [
    ("pack", "ab60aea8b93094e59865b323f9ee26953ef4737a0b069078afe692dd7c33e0e6"),
    ("stack", "b62cb3acb0b3c7074553823cdef89d4bf37d4913b0d486fcb16cb4c5a845ddc6"),
])
def test_synthetic_startup_observation_is_deterministic_and_private_data_free(task, expected_sha256):
    config = yaml.safe_load((ROOT / "configs" / f"robot_{task}_w4a8.example.yaml").read_text())
    first = synthetic_audit_observation(config)
    second = synthetic_audit_observation(config)
    assert first["image_channel_order"] == "native"
    assert first["state"].shape == (14,)
    assert np.isfinite(first["state"]).all()
    assert set(first["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    for camera in first["images"]:
        assert first["images"][camera].shape == (480, 640, 3)
        assert first["images"][camera].dtype == np.uint8
        np.testing.assert_array_equal(first["images"][camera], second["images"][camera])
    assert audit_observation_sha256(first) == audit_observation_sha256(second) == expected_sha256


def test_upstream_lock_matches_runtime():
    lock = json.loads((ROOT / "third_party" / "versions.lock.json").read_text())
    assert lock["fastwam"]["commit"] == FASTWAM_REVISION
