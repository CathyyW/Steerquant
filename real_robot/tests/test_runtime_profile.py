import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastwam_steerquant import runtime_profile as runtime_profile
from fastwam_steerquant.kernels.w4a8 import loader


PROFILE = Path(__file__).resolve().parents[1] / "configs" / "real_robot_w4a8_runtime.json"


@pytest.fixture
def clean_runtime(monkeypatch):
    for name in (
        "FASTWAM_RUNTIME_PROFILE",
        "FASTWAM_W4A8_TILE",
        "FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime_profile, "_configured_tile", None)
    monkeypatch.setattr(
        loader,
        "load_extension",
        SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=0)),
    )


def test_release_profile_enables_audited_runtime(clean_runtime):
    options, profile = runtime_profile.resolve_runtime_profile(profile=PROFILE)
    assert options == {"construct_device": "cpu", "cuda_graph": True, "fuse_block": False}
    assert profile["w4a8_tile"] == 64


def test_release_profile_rejects_fusion(clean_runtime, tmp_path):
    payload = json.loads(PROFILE.read_text())
    payload["fuse_block"] = True
    invalid = tmp_path / "profile.json"
    invalid.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no block fusion"):
        runtime_profile.resolve_runtime_profile(profile=invalid)
