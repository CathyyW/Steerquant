from collections import Counter
import pytest
from steerquant.integrations.robocasa_protocol import make_manifest, normalize_metadata, summarize


def test_original_ids_cover_all_scenes_with_two_trials_per_seed_task():
    m = make_manifest()
    assert len(m["episodes"]) == len({e["key"] for e in m["episodes"]}) == 720
    cells = Counter((e["task"], e["policy_seed"], e["scene_index"]) for e in m["episodes"])
    assert len(cells) == 24 * 3 * 5
    assert set(cells.values()) == {2}
    for e in m["episodes"]:
        assert e["scene_index"] == e["episode_id"] // 10
        assert e["env_seed"] == e["policy_seed"] * e["episode_id"] * 256


def test_summary_reports_sample_sd_only_after_all_seeds_complete():
    m = make_manifest()
    results = [{**e, "success": e["policy_seed"] != 196} for e in m["episodes"]]
    summary = summarize(m, results)
    assert summary["complete"] and summary["successes"] == 480
    assert summary["mean_success_rate"] == pytest.approx(2 / 3)
    assert summary["seed_sample_sd"] == pytest.approx(3**-0.5)
    partial = summarize(m, results[:240])
    assert not partial["complete"] and partial["mean_success_rate"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(m, results + results[:1])
    results[0] = {**results[0], "scene_index": 1}
    with pytest.raises(ValueError, match="manifest"):
        summarize(m, results)


def test_metadata_normalization_preserves_language_and_physical_order():
    a = dict(lang="stop microwave", objects=[dict(groups_containing_sampled_obj=["cup", "all"], pos=[2, 1])])
    b = dict(lang="stop microwave", objects=[dict(groups_containing_sampled_obj=["all", "cup"], pos=[2, 1])])
    assert normalize_metadata(a) == normalize_metadata(b)
    b["objects"][0]["pos"] = [1, 2]
    assert normalize_metadata(a) != normalize_metadata(b)
    b["objects"][0]["pos"] = [2, 1]
    b["lang"] = "start microwave"
    assert normalize_metadata(a) != normalize_metadata(b)


def test_shared_state_json_roundtrip_and_real_mismatch(tmp_path):
    from steerquant.evaluate_robocasa import shared_json

    path = tmp_path / "state.json"
    shared_json(path, {"camera": (1, 2), "state_sha256": "same"})
    shared_json(path, {"camera": (1, 2), "state_sha256": "same"})
    with pytest.raises(RuntimeError, match="mismatch"):
        shared_json(path, {"camera": (1, 2), "state_sha256": "different"})
