from __future__ import annotations
import csv
from types import SimpleNamespace
import torch
import torch.nn as nn
from steerquant.algorithms.wam_d import WAMDCalibrationConfig
from steerquant.algorithms.wam_gamma import WAMGammaCalibrationConfig
from steerquant.runtime import quantize_model_for_inference
from steerquant.wam_cache import WAMActivationCache, WAMCacheConfig
from steerquant.wam_pipeline import (
    WAMCalibrationCheckpoint,
    WAMNetworkCallTracker,
    WAMSensitivityField,
    build_wam_cache_sites,
    calibrate_wam_model,
    calibrate_wam_model_grouped,
    load_wam_sensitivity_field_csv,
    replace_linears_with_wam,
)
from steerquant.wam_runtime import WAMFakeQuantLinear


class _PipelineNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[1](torch.tanh(self.layers[0](x)))


class _WrappedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _CheckpointWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = _WrappedLinear()


def _pipeline_inputs() -> tuple[_PipelineNet, WAMActivationCache, WAMSensitivityField]:
    torch.manual_seed(81)
    model = _PipelineNet()
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=2,
            num_denoise_calls=2,
            num_streams=3,
            max_observations=2,
            tokens_per_observation=2,
            storage_dtype="float32",
        )
    )
    generator = torch.Generator().manual_seed(82)
    for observation in range(2):
        for timestep in range(2):
            for stream in range(3):
                cache.add(
                    layer=0,
                    timestep=timestep,
                    stream=stream,
                    observation_id=observation,
                    module_name="layers.0",
                    values=torch.randn(3 + stream, 4, generator=generator),
                )
            cache.add(
                layer=1,
                timestep=timestep,
                stream=0,
                observation_id=observation,
                module_name="layers.1",
                values=torch.randn(5, 4, generator=generator),
            )
    mask = torch.tensor([[[1, 1, 1], [1, 1, 1]], [[1, 0, 0], [1, 0, 0]]], dtype=torch.bool)
    field = WAMSensitivityField(
        values=torch.tensor([[[1.0, 2.0, 4.0], [4.0, 2.0, 1.0]], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        active_stream_mask=mask,
        module_names=("layers.0", "layers.1"),
        slot_aligned=(True, False),
        observation_counts=mask.long() * 2,
    )
    return (model, cache, field)


def _calibrate_small_checkpoint() -> tuple[_PipelineNet, WAMCalibrationCheckpoint]:
    model, cache, field = _pipeline_inputs()
    checkpoint = calibrate_wam_model(
        model,
        cache=cache,
        sensitivity_field=field,
        rho=0.5,
        d_config=WAMDCalibrationConfig(weight_bits=4, activation_bits=4, epochs=1, batch_size=4),
        gamma_config=WAMGammaCalibrationConfig(
            activation_bits=4, epochs=2, batch_size=4, gamma_min=0.5, gamma_max=2.0
        ),
    )
    return (model, checkpoint)


def test_sensitivity_csv_uses_root_mean_energy_and_global_stream_mask(tmp_path) -> None:
    path = tmp_path / "scores.csv"
    columns = [
        "observation_id",
        "block_index",
        "linear_index",
        "module_path",
        "denoise_call",
        "slot_aligned",
        "slot_index",
        "quantization_variant",
        "predicted_standardized_action_rmse",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for observation, aligned_score, global_score in ((0, 1.0, 2.0), (1, 3.0, 4.0)):
            for timestep in range(2):
                for stream in range(2):
                    writer.writerow(
                        {
                            "observation_id": observation,
                            "block_index": 1,
                            "linear_index": 0,
                            "module_path": "self_attn.q_proj",
                            "denoise_call": timestep,
                            "slot_aligned": True,
                            "slot_index": stream,
                            "quantization_variant": "joint_wa",
                            "predicted_standardized_action_rmse": aligned_score,
                        }
                    )
                writer.writerow(
                    {
                        "observation_id": observation,
                        "block_index": 1,
                        "linear_index": 1,
                        "module_path": "cross_attn.k_proj",
                        "denoise_call": timestep,
                        "slot_aligned": False,
                        "slot_index": -1,
                        "quantization_variant": "joint_wa",
                        "predicted_standardized_action_rmse": global_score,
                    }
                )
    field = load_wam_sensitivity_field_csv(
        path, num_denoise_calls=2, num_semantic_streams=2, expected_observations=2
    )
    torch.testing.assert_close(field.values[0], torch.full((2, 2), 5.0**0.5))
    torch.testing.assert_close(field.values[1, :, 0], torch.full((2,), 10.0**0.5))
    assert torch.equal(field.active_stream_mask[1], torch.tensor([[1, 0], [1, 0]], dtype=torch.bool))
    assert field.module_names == ("blocks.0.self_attn.q_proj", "blocks.0.cross_attn.k_proj")


def test_cache_sites_resolve_checkpoint_wrapped_module_names() -> None:
    model = nn.Module()
    model.blocks = nn.ModuleList([_CheckpointWrapper()])
    field = WAMSensitivityField(
        values=torch.ones(1, 1, 1),
        active_stream_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        module_names=("blocks.0.linear",),
        slot_aligned=(False,),
        observation_counts=torch.ones(1, 1, 1, dtype=torch.long),
    )
    sites = build_wam_cache_sites(model, field)
    assert len(sites) == 1
    assert sites[0].module is model.blocks[0]._checkpoint_wrapped_module.linear
    assert sites[0].module_name == "blocks.0.linear"


def test_end_to_end_calibration_checkpoint_runtime_and_global_context(tmp_path) -> None:
    model, checkpoint = _calibrate_small_checkpoint()
    assert checkpoint.weight_bits == checkpoint.activation_bits == 4
    assert len(checkpoint.linears) == 2
    assert checkpoint.linears[0].stream_gains.shape == (2, 3)
    assert checkpoint.linears[0].stream_indices == (0, 1, 2)
    assert checkpoint.linears[1].stream_gains.shape == (2, 1)
    assert checkpoint.linears[1].stream_indices == (0,)
    torch.testing.assert_close(checkpoint.linears[1].stream_gains, torch.ones(2, 1))
    assert all((entry.d_final_hard_loss <= entry.d_initial_hard_loss for entry in checkpoint.linears))
    assert all(
        (
            final <= initial
            for entry in checkpoint.linears
            for initial, final in zip(entry.gamma_initial_hard_losses, entry.gamma_final_hard_losses)
        )
    )
    path = checkpoint.save(tmp_path / "wam.pt")
    restored = WAMCalibrationCheckpoint.load(path)
    assert restored.d_config["weight_bits"] == 4
    torch.testing.assert_close(restored.sensitivity, checkpoint.sensitivity)
    source_device = next(model.parameters()).device
    assert replace_linears_with_wam(model, restored) == 2
    assert isinstance(model.layers[0], WAMFakeQuantLinear)
    assert isinstance(model.layers[1], WAMFakeQuantLinear)
    assert model.layers[0].qweight.device == source_device
    tracker = WAMNetworkCallTracker(model, restored.num_denoise_calls)
    tracker.register()
    try:
        outputs = [model(torch.randn(1, 6, 4)) for _ in range(4)]
    finally:
        tracker.remove()
    assert tracker.total_calls == 4
    assert all((output.shape == (1, 6, 4) and torch.isfinite(output).all() for output in outputs))


def test_grouped_calibration_requests_and_combines_one_cache_group_at_a_time() -> None:
    model, cache, field = _pipeline_inputs()
    requested: list[tuple[int, ...]] = []

    def provide(group: tuple[int, ...]) -> WAMActivationCache:
        requested.append(group)
        return cache

    checkpoint = calibrate_wam_model_grouped(
        model,
        sensitivity_field=field,
        cache_provider=provide,
        layers_per_group=1,
        d_config=WAMDCalibrationConfig(weight_bits=4, activation_bits=4, epochs=0),
        gamma_config=WAMGammaCalibrationConfig(activation_bits=4, epochs=0),
    )
    assert requested == [(0,), (1,)]
    assert [entry.layer for entry in checkpoint.linears] == [0, 1]


def test_full_calibration_can_release_each_layer_cache_after_use() -> None:
    model, cache, field = _pipeline_inputs()
    checkpoint = calibrate_wam_model(
        model,
        cache=cache,
        sensitivity_field=field,
        d_config=WAMDCalibrationConfig(weight_bits=4, activation_bits=4, epochs=0),
        gamma_config=WAMGammaCalibrationConfig(activation_bits=4, epochs=0),
        release_cache_after_layer=True,
    )
    assert len(checkpoint.linears) == 2
    assert cache.keys == ()
    assert cache.storage_nbytes == 0


def test_grouped_calibration_matches_full_and_resumes_without_replaying_saved_layers() -> None:
    model, cache, field = _pipeline_inputs()
    configs = dict(
        sensitivity_field=field, rho=0.5,
        d_config=WAMDCalibrationConfig(weight_bits=4, activation_bits=8, epochs=2, batch_size=4),
        gamma_config=WAMGammaCalibrationConfig(activation_bits=8, epochs=2, batch_size=4),
    )
    reference = calibrate_wam_model(model, cache=cache, **configs)
    requested, saved = [], []

    def provide(group):
        requested.append(group)
        _, fresh, _ = _pipeline_inputs()
        for layer in range(2):
            if layer not in group:
                fresh.release_layer(layer)
        return fresh

    grouped = calibrate_wam_model_grouped(
        model, cache_provider=provide, layers_per_group=1, layer_completed=saved.append, **configs,
    )
    assert requested == [(0,), (1,)]
    assert len(saved) == 2
    requested.clear()
    resumed = calibrate_wam_model_grouped(
        model, cache_provider=provide, layers_per_group=2, completed_linears={0: saved[0]}, **configs,
    )
    assert requested == [(1,)]
    for candidate in [grouped, resumed]:
        for expected, actual in zip(reference.linears, candidate.linears):
            for name in ['qweight', 'weight_scales', 'input_scale', 'stream_gains',
                         'clipping_ranges', 'token_fractions']:
                torch.testing.assert_close(getattr(expected, name), getattr(actual, name), rtol=0, atol=0)


def test_reconstruction_group_replay_preserves_sampling_and_global_indices(monkeypatch) -> None:
    from steerquant import calibration
    model, _, field = _pipeline_inputs()
    args = SimpleNamespace(num_denoising_steps=2, num_streams=3, tokens_per_observation=2,
                           sampling_seed=19, storage_dtype='float16', observation_start=7,
                           seed=195, progress_every=10)
    records = [{'observation': i, 'task_label': str(i)} for i in range(2)]

    def batch(_cfg, _model, _stats, obs, _label):
        return torch.randn(1, 6, 4, generator=torch.Generator().manual_seed(obs)), None

    def generate(wrapper, data, **kwargs):
        for _ in range(2):
            result = wrapper.net(data)
        return result

    monkeypatch.setattr(calibration, 'build_policy_data_batch', batch)
    monkeypatch.setattr(calibration, 'generate_matched', generate)
    wrapper = SimpleNamespace(net=model)
    full = calibration._collect_cache(args, wrapper, records, {}, None, field)
    grouped = calibration._collect_cache(args, wrapper, records, {}, None, field, (1,))
    assert all(key.layer == 1 for key in grouped.keys)
    assert grouped.storage_nbytes < full.storage_nbytes
    assert grouped.accepted_observations == full.accepted_observations == (7, 8)
    for key in grouped.keys:
        coords = (key.layer, key.timestep, key.stream)
        assert torch.equal(full.activations(*coords), grouped.activations(*coords))
        assert torch.equal(full.channel_absmax(*coords), grouped.channel_absmax(*coords))
        assert torch.equal(full.row_observation_ids(*coords), grouped.row_observation_ids(*coords))
        assert full.num_seen_tokens(*coords) == grouped.num_seen_tokens(*coords)


def test_generic_runtime_loads_wam_checkpoint_and_tracks_calls(tmp_path) -> None:
    _calibrated_model, checkpoint = _calibrate_small_checkpoint()
    path = checkpoint.save(tmp_path / "wam.pt")
    model, _cache, _field = _pipeline_inputs()
    wrapper = SimpleNamespace(net=model)
    runtime = SimpleNamespace(
        quantization_enabled=True,
        quantization_algo="wam",
        quantization_weight_bits=4,
        quantization_activation_bits=4,
        quantization_calibration_path=str(path),
    )
    quantize_model_for_inference(wrapper, runtime)
    assert isinstance(model.layers[0], WAMFakeQuantLinear)
    assert isinstance(model.layers[1], WAMFakeQuantLinear)
    assert hasattr(model, "_wam_network_call_tracker")
    first = model(torch.randn(1, 6, 4))
    second = model(torch.randn(1, 6, 4))
    assert first.shape == second.shape == (1, 6, 4)
    assert model._wam_network_call_tracker.total_calls == 2


def test_preprocess_signs_survive_wam_checkpoint_and_runtime(tmp_path) -> None:
    model, cache, field = _pipeline_inputs()
    rotations = {
        "layers.0": torch.tensor([1.0, -1.0, 1.0, -1.0]),
        "layers.1": torch.tensor([-1.0, 1.0, -1.0, 1.0]),
    }
    checkpoint = calibrate_wam_model(
        model,
        cache=cache,
        sensitivity_field=field,
        d_config=WAMDCalibrationConfig(weight_bits=4, activation_bits=4, epochs=0),
        gamma_config=WAMGammaCalibrationConfig(activation_bits=4, epochs=0),
        input_rotation_signs=rotations,
    )

    restored = WAMCalibrationCheckpoint.load(checkpoint.save(tmp_path / "preprocess_wam.pt"))
    for entry in restored.linears:
        torch.testing.assert_close(entry.input_rotation_signs, rotations[entry.module_name])

    assert replace_linears_with_wam(model, restored) == 2
    assert all(module.input_rotation_signs is not None for module in model.layers)
    tracker = WAMNetworkCallTracker(model, restored.num_denoise_calls)
    tracker.register()
    try:
        output = model(torch.randn(1, 6, 4))
    finally:
        tracker.remove()
    assert output.shape == (1, 6, 4)
    assert torch.isfinite(output).all()
