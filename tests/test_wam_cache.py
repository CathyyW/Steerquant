from __future__ import annotations

import torch
import torch.nn as nn

from steerquant.wam_cache import (
    WAMActivationCache,
    WAMActivationCacheCollector,
    WAMCacheConfig,
    WAMCacheSite,
    split_activation_streams,
)


def test_split_activation_streams_preserves_slot_rows() -> None:
    grid = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
    flattened = grid.reshape(2, 6, 4)

    grid_streams = split_activation_streams(grid, num_streams=3, slot_aligned=True)
    flat_streams = split_activation_streams(flattened, num_streams=3, slot_aligned=True)

    assert len(grid_streams) == 3
    for stream_index in range(3):
        torch.testing.assert_close(grid_streams[stream_index], grid[:, stream_index])
        torch.testing.assert_close(flat_streams[stream_index], grid[:, stream_index])


def test_cache_balances_observations_and_respects_cap() -> None:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=1,
            num_denoise_calls=1,
            num_streams=1,
            max_observations=2,
            tokens_per_observation=1,
        )
    )

    assert cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=0,
        module_name="linear",
        values=torch.arange(20, dtype=torch.float32).reshape(5, 4),
    )
    assert not cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=0,
        module_name="linear",
        values=torch.ones(5, 4),
    )
    assert cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=1,
        module_name="linear",
        values=torch.ones(7, 4),
    )
    assert not cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=2,
        module_name="linear",
        values=torch.ones(9, 4),
    )

    assert cache.activations(0, 0, 0).shape == (2, 4)
    assert cache.activations(0, 0, 0).dtype == torch.float16
    assert cache.row_observation_ids(0, 0, 0).tolist() == [0, 1]
    assert cache.num_seen_tokens(0, 0, 0) == 12
    assert cache.accepted_observations == (0, 1)


def test_cache_uses_full_token_counts_for_gain_conservation() -> None:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=1,
            num_denoise_calls=1,
            num_streams=2,
            tokens_per_observation=1,
        )
    )
    cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=0,
        module_name="linear",
        values=torch.ones(2, 4),
    )
    cache.add(
        layer=0,
        timestep=0,
        stream=1,
        observation_id=0,
        module_name="linear",
        values=torch.ones(6, 4),
    )

    torch.testing.assert_close(cache.token_fractions(0, 0), torch.tensor([0.25, 0.75]))
    assert cache.activations(0, 0, 0).shape[0] == cache.activations(0, 0, 1).shape[0] == 1


def test_cache_collector_hooks_linear_inputs_by_timestep_and_stream() -> None:
    linear = nn.Linear(4, 4, bias=False)
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=1,
            num_denoise_calls=2,
            num_streams=3,
            tokens_per_observation=1,
        )
    )
    collector = WAMActivationCacheCollector(
        cache,
        [
            WAMCacheSite(
                layer=0,
                module_name="linear",
                module=linear,
                slot_aligned=True,
                stream_indices=(0, 1, 2),
            )
        ],
    )
    collector.register()
    try:
        assert collector.begin_observation(7)
        collector.begin_denoise_call(1)
        linear(torch.randn(1, 6, 4))
    finally:
        collector.remove()

    for stream in range(3):
        assert cache.activations(0, 1, stream).shape == (1, 4)
        assert cache.num_seen_tokens(0, 1, stream) == 2


def test_cache_defaults_to_16_tokens_from_320_observations() -> None:
    config = WAMCacheConfig(num_layers=1, num_denoise_calls=1, num_streams=1)
    cache = WAMActivationCache(config)

    assert config.max_observations == 320
    assert config.tokens_per_observation == 16
    cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=0,
        module_name="linear",
        values=torch.arange(80, dtype=torch.float32).reshape(20, 4),
    )

    assert cache.activations(0, 0, 0).shape == (16, 4)
    assert cache.row_observation_ids(0, 0, 0).tolist() == [0] * 16
    assert cache.num_seen_tokens(0, 0, 0) == 20


def test_sampled_cache_preserves_full_token_channel_absmax() -> None:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=1,
            num_denoise_calls=1,
            num_streams=1,
            tokens_per_observation=1,
        )
    )
    values = torch.tensor([[1.0, -20.0], [30.0, 2.0], [3.0, 4.0]])
    cache.add(
        layer=0,
        timestep=0,
        stream=0,
        observation_id=0,
        module_name="linear",
        values=values,
    )

    assert cache.activations(0, 0, 0).shape == (1, 2)
    torch.testing.assert_close(cache.channel_absmax(0, 0, 0), torch.tensor([30.0, 20.0]))


def test_cache_roundtrip_preserves_samples_and_metadata(tmp_path) -> None:
    cache = WAMActivationCache(WAMCacheConfig(num_layers=2, num_denoise_calls=2, num_streams=2))
    cache.add(
        layer=1,
        timestep=0,
        stream=1,
        observation_id=4,
        module_name="blocks.0.mlp.layer1",
        values=torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )

    path = cache.save(tmp_path / "wam_cache.pt")
    restored = WAMActivationCache.load(path)

    assert restored.config == cache.config
    assert restored.keys == cache.keys
    assert restored.accepted_observations == (4,)
    assert restored.num_seen_tokens(1, 0, 1) == 3
    torch.testing.assert_close(restored.channel_absmax(1, 0, 1), cache.channel_absmax(1, 0, 1))
    torch.testing.assert_close(restored.activations(1, 0, 1), cache.activations(1, 0, 1))
    torch.testing.assert_close(
        restored.row_observation_ids(1, 0, 1),
        cache.row_observation_ids(1, 0, 1),
    )


def test_cache_reports_storage_and_can_release_one_layer() -> None:
    cache = WAMActivationCache(
        WAMCacheConfig(
            num_layers=2,
            num_denoise_calls=1,
            num_streams=1,
            max_observations=1,
            tokens_per_observation=2,
        )
    )
    for layer in range(2):
        cache.add(
            layer=layer,
            timestep=0,
            stream=0,
            observation_id=0,
            module_name=f"blocks.{layer}.linear",
            values=torch.randn(3, 4),
        )

    before = cache.storage_nbytes
    assert before > 0
    assert cache.release_layer(0) == 1
    assert all(key.layer == 1 for key in cache.keys)
    assert 0 < cache.storage_nbytes < before
