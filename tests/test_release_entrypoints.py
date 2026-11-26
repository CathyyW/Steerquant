import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from steerquant.calibrate import build_parser as calibration_parser, resolve_defaults
from steerquant.evaluate import build_parser as libero_parser
from steerquant import evaluate_robocasa as robocasa
from steerquant.integrations import cosmos


@pytest.mark.parametrize("suite,expected", [("libero", 320), ("robocasa", 192)])
def test_platform_calibration_defaults_and_explicit_override(suite, expected):
    argv = [
        "--suite",
        suite,
        "--model-dir",
        "model",
        "--tokenizer",
        "tokenizer",
        "--calibration",
        "samples.pkl",
        "--output-dir",
        "out",
    ]
    assert resolve_defaults(calibration_parser().parse_args(argv)).observations == expected
    assert resolve_defaults(calibration_parser().parse_args(argv + ["--observations", "2"])).observations == 2


@pytest.mark.parametrize("task_suite", ["libero_spatial", "libero_object", "libero_goal", "libero_10"])
def test_libero_task_suite_is_separate_from_platform(tmp_path, monkeypatch, task_suite):
    args = libero_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path / "tokenizer.pth"),
            "--checkpoint",
            str(tmp_path / "quant.pt"),
            "--suite",
            task_suite,
        ]
    )
    args.suite = "libero"
    for name in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_dataset_statistics.json",
        "libero_t5_embeddings.pkl",
        "tokenizer.pth",
    ):
        (tmp_path / name).touch()
    monkeypatch.setattr(cosmos, "configure_platform", lambda suite: None)
    monkeypatch.setattr(
        cosmos.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path / "installed")]),
    )
    cosmos.prepare_paths(args)
    assert args.task_suite == task_suite
    assert args.checkpoint.name == "Cosmos-Policy-LIBERO-Predict2-2B.pt"
    assert Path(args.config_file).is_absolute()


def test_platform_constants_ignore_misleading_paths_and_restore_argv(monkeypatch):
    argv = ["steerquant-calibrate", "--suite", "robocasa", "--model-dir", "/libero/robocasa"]
    monkeypatch.setattr(sys, "argv", argv)

    def import_constants(name):
        assert name == "cosmos_policy.constants"
        assert sys.argv == ["steerquant", "--suite", "robocasa"]
        return SimpleNamespace(ROBOT_PLATFORM="ROBOCASA", NUM_ACTIONS_CHUNK=32)

    monkeypatch.setattr(cosmos.importlib, "import_module", import_constants)
    cosmos.configure_platform("robocasa")
    assert sys.argv is argv
    monkeypatch.setattr(
        cosmos.importlib,
        "import_module",
        lambda name: SimpleNamespace(ROBOT_PLATFORM="LIBERO", NUM_ACTIONS_CHUNK=16),
    )
    with pytest.raises(RuntimeError, match="fresh process"):
        cosmos.configure_platform("robocasa")
    assert sys.argv is argv


def test_robocasa_requires_explicit_matching_checkpoint(tmp_path):
    argv = ["--mode", "smoke", "--model-dir", "model", "--tokenizer", "tokenizer"]
    with pytest.raises(ValueError, match="required"):
        robocasa.validate_args(robocasa.build_parser().parse_args(argv + ["--method", "w4a8"]))
    with pytest.raises(ValueError, match="does not accept"):
        robocasa.validate_args(
            robocasa.build_parser().parse_args(argv + ["--method", "fp", "--checkpoint", "q.pt"])
        )
    checkpoint = tmp_path / "quant.pt"
    checkpoint.touch()
    args = robocasa.validate_args(
        robocasa.build_parser().parse_args(
            argv + ["--method", "w4a4", "--checkpoint", str(checkpoint), "--no-cuda-graph"]
        )
    )
    assert args.quant_checkpoint == checkpoint
    assert args.suite == "robocasa" and not args.cuda_graph


@pytest.mark.parametrize("method", ["fp", "w4a8", "w4a4"])
def test_robocasa_config_preserves_research_protocol(method):
    args = SimpleNamespace(
        method=method,
        config="experiment",
        config_file="/installed/config.py",
        checkpoint=Path("/model.pt"),
        dataset_stats=Path("/stats.json"),
        t5_embeddings=Path("/t5.pkl"),
        quant_checkpoint=Path("/quant.pt"),
    )
    cfg = robocasa.make_config(args, SimpleNamespace(PolicyEvalConfig=SimpleNamespace))
    assert (cfg.chunk_size, cfg.num_open_loop_steps, cfg.num_denoising_steps_action) == (32, 16, 5)
    assert (cfg.num_third_person_images, cfg.num_wrist_images) == (2, 1)
    assert cfg.use_jpeg_compression and cfg.flip_images and cfg.deterministic_reset
    assert not cfg.ar_future_prediction and not cfg.ar_value_prediction and not cfg.dit_cuda_graph
    assert cfg.quantization_enabled == (method != "fp")
    if method != "fp":
        assert cfg.quantization_backend == "cutlass"
        assert cfg.quantization_activation_bits == int(method[-1])


def test_robocasa_action_checker_rejects_bad_shapes_and_schedule():
    import numpy as np

    tracker = SimpleNamespace(total_calls=0, validate_complete_schedule=lambda: None)
    model = SimpleNamespace(net=SimpleNamespace(_wam_network_call_tracker=tracker))

    def action(calls=5, shape=(32, 7)):
        tracker.total_calls += calls
        return {"actions": np.zeros(shape)}

    assert robocasa.checked_action(action, model, "w4a8")["actions"].shape == (32, 7)
    with pytest.raises(RuntimeError):
        robocasa.checked_action(lambda: action(calls=4), model, "w4a8")
    with pytest.raises(RuntimeError):
        robocasa.checked_action(lambda: action(shape=(16, 7)), model, "w4a8")


def test_robocasa_loader_installs_release_fusions_and_runtime(monkeypatch):
    from steerquant import runtime

    calls = []
    model = SimpleNamespace(net=object())
    monkeypatch.setattr(cosmos, "get_model", lambda cfg: (model, "model-config"))
    monkeypatch.setitem(
        sys.modules,
        "steerquant.integrations.fusions",
        SimpleNamespace(install_fusions=lambda net: calls.append(("fusions", net))),
    )
    monkeypatch.setattr(
        runtime, "quantize_model_for_inference", lambda model, cfg: calls.append(("runtime", model))
    )
    assert robocasa.load_model(SimpleNamespace(quantization_enabled=True)) == (model, "model-config")
    assert calls == [("fusions", model.net), ("runtime", model)]


@pytest.mark.parametrize("suite", ["libero", "robocasa"])
def test_model_loader_uses_local_tokenizer_and_restores_checkpoint_resolver(monkeypatch, suite):
    original = lambda uri, *a, **kw: "resolved:" + uri
    checkpoint_db = SimpleNamespace(get_checkpoint_path=original)
    monkeypatch.setitem(
        sys.modules, "cosmos_policy._src.imaginaire.utils", SimpleNamespace(checkpoint_db=checkpoint_db)
    )
    monkeypatch.setenv("COSMOS_POLICY_TOKENIZER", "/assets/local-tokenizer.pth")
    calls = []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            calls.append(device)
            return self

    model = Model()

    def loader(**kwargs):
        calls.append(kwargs)
        assert (
            checkpoint_db.get_checkpoint_path(
                "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
            )
            == "/assets/model.pt"
        )
        assert checkpoint_db.get_checkpoint_path("unrelated") == "resolved:unrelated"
        return model, "config"

    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy._src.predict2.utils.model_loader",
        SimpleNamespace(load_model_from_checkpoint=loader),
    )
    cfg = SimpleNamespace(
        suite=suite,
        ckpt_path="/assets/model.pt",
        config="experiment",
        config_file="/installed/cosmos_policy/config/config.py",
    )
    assert cosmos.get_model(cfg) == (model, "config")
    assert calls[0]["experiment_opts"] == ["model.config.tokenizer.vae_pth=/assets/local-tokenizer.pth"]
    assert calls[0]["config_file"] == cfg.config_file
    assert calls[1] == "cuda"
    assert checkpoint_db.get_checkpoint_path is original
