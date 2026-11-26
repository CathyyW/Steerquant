from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from steerquant.integrations.linear_sites import LinearSite
from steerquant.map_reuse import ReusableRecomputeContext, retained_backward_state, run_action_probes
from steerquant.sensitivity import FirstOrderLinearCollector


@pytest.mark.parametrize("mode", ["reuse", "replay"])
def test_calibration_has_no_forward_mode_switch(mode):
    from steerquant.calibrate import build_parser

    argv = [
        "--model-dir",
        "model",
        "--tokenizer",
        "tokenizer",
        "--calibration",
        "samples.pkl",
        "--output-dir",
        "out",
        "--map-forward-mode",
        mode,
    ]
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(argv)
    assert error.value.code == 2


class TinyMapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.aligned = nn.Linear(4, 4)
        self.context = nn.Linear(4, 4)
        self.register_buffer("text", torch.randn(1, 2, 4))

    def forward(self, x, sigma):
        return torch.tanh(self.aligned(x) + self.context(self.text).mean(1, keepdim=True)) / (1 + sigma)


def sites(net):
    return [
        LinearSite(1, 0, "aligned", "aligned", net.aligned, True),
        LinearSite(1, 1, "context", "context", net.context, False),
    ]


def run_replay_reference(build_forward, collector, generator, *, on_probe):
    """Test-only oracle for the previous per-probe sampling algorithm."""
    for probe in range(collector.num_action_probes):
        collector.begin_pass()
        action = build_forward()
        collector.validate_capture_count()
        signs = torch.randint(0, 2, action.shape, generator=generator).float().mul_(2).sub_(1)
        collector.set_probe(probe)
        (action * signs).sum().backward()
        collector.validate_backward_count()
        on_probe(probe, 0)
    return collector.num_action_probes


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("streams", [9, 11])
@pytest.mark.parametrize("bits,rotation_seed", [(8, None), (4, 1234)])
def test_one_forward_matches_sixteen_replays_and_releases_last_probe(
    monkeypatch, checkpoint, streams, bits, rotation_seed
):
    monkeypatch.setattr("steerquant.sensitivity.enumerate_block_linears", sites)
    torch.manual_seed(45)
    template = TinyMapNet().eval().requires_grad_(False)
    initial = torch.randn(1, streams * 2, 4)

    def evaluate(reuse):
        net = deepcopy(template)
        collector = FirstOrderLinearCollector(
            net,
            weight_bits=4,
            activation_bits=bits,
            activation_granularity="per_tensor",
            activation_clipping_ranges={
                "blocks.0.aligned": torch.ones(5) * 3,
                "blocks.0.context": torch.ones(5) * 3,
            },
            num_action_probes=16,
            num_action_elements=initial.numel(),
            num_denoising_steps=5,
            slot_names=tuple(f"s{i}" for i in range(streams)),
            checkpoint_denoisers=checkpoint,
            rotation_seed=rotation_seed,
            reuse_forward=reuse,
        )
        collector.register()
        all_results = []
        try:
            # A second observation checks that retained captures/probe totals do not leak.
            for observation in range(2):
                collector.reset_observation()
                gradients, outputs, seen = [], [], []

                def forward():
                    leaf = (initial + observation / 10).clone().requires_grad_(True)
                    leaf.register_hook(lambda grad: gradients.append(grad.detach().clone()))
                    x = leaf
                    history = x * 0.1
                    for call in range(5):
                        sigma = torch.tensor(0.1 * call)
                        collector.begin_denoise_call(call, sigma)
                        y = collector.checkpoint_call(net, x, sigma, call) if checkpoint else net(x, sigma)
                        x = x * 0.3 + y + history * 0.1
                        history = y
                    outputs.append(x.detach().clone())
                    return x

                def after_probe(index, seconds):
                    seen.append(len(collector.backward_seen))
                    assert collector.tracker.total_calls == 5
                    if checkpoint or not reuse or index == 15:
                        assert all(c.inputs is None and c.output is None for c in collector.captures)
                    else:
                        assert all(c.inputs is not None and c.output is not None for c in collector.captures)

                probe_runner = run_action_probes if reuse else run_replay_reference
                forwards = probe_runner(
                    forward,
                    collector,
                    torch.Generator().manual_seed(100 + observation),
                    on_probe=after_probe,
                )
                assert forwards == len(outputs) == (1 if reuse else 16)
                assert len(gradients) == 16 and seen == [10] * 16
                rows = collector.export_action_probe_rows({"observation_id": f"test-{observation}"})
                assert len(rows) == (streams + 1) * 5
                all_results.append((outputs, gradients, rows))
            return all_results
        finally:
            collector.remove()

    baseline, reused = evaluate(False), evaluate(True)
    for reference, actual in zip(baseline, reused):
        for output in reference[0]:
            torch.testing.assert_close(output, actual[0][0], rtol=0, atol=0)
        for expected, observed in zip(reference[1], actual[1]):
            torch.testing.assert_close(expected, observed, rtol=0, atol=0)
        assert reference[2] == actual[2]


class ConsumingContext:
    def __init__(self):
        self.storage = {"mm": [10, 20]}
        counters = {"mm": 0}

        def policy():
            counters["mm"] += 1
            return counters["mm"]

        self.policy_fn = policy

    def __enter__(self):
        return self.storage["mm"].pop(0), self.policy_fn()

    def __exit__(self, *exc):
        return False


def test_selective_checkpoint_recompute_resets_consumed_cache_and_policy_counters():
    context = ReusableRecomputeContext(ConsumingContext())
    for _ in range(16):
        with context as value:
            assert value == (10, 1)


@pytest.mark.parametrize("fail", [False, True])
def test_backward_compatibility_preserves_metadata_and_restores_all_patches(monkeypatch, fail):
    rmsnorm = SimpleNamespace(clear_tensor_data=lambda tensor: tensor.zero_())

    class Fuser:
        @staticmethod
        def backward(ctx, *args):
            for op in ctx.basic_op_ctxs:
                del op._saved_tensors_range
                rmsnorm.clear_tensor_data(op.rstdevs)
            return "gradient"

    original_backward = Fuser.backward
    original_clear = rmsnorm.clear_tensor_data
    original_context = lambda: (nullcontext(), ConsumingContext())
    final = SimpleNamespace(
        _checkpoint_wrapped_module=object(),
        checkpoint_fn=partial(lambda **kw: None, context_fn=original_context),
    )
    net = SimpleNamespace(final_layer=final)
    monkeypatch.setattr("steerquant.map_reuse._load_transformer_engine", lambda: (Fuser, rmsnorm))
    ctx = SimpleNamespace(basic_op_ctxs=[SimpleNamespace(_saved_tensors_range=(0, 1), rstdevs=torch.ones(2))])
    try:
        with retained_backward_state(net, enabled=True):
            for _ in range(16):
                assert Fuser.backward(ctx) == "gradient"
                assert ctx.basic_op_ctxs[0]._saved_tensors_range == (0, 1)
                assert torch.equal(ctx.basic_op_ctxs[0].rstdevs, torch.ones(2))
            _, recompute = final.checkpoint_fn.keywords["context_fn"]()
            assert isinstance(recompute, ReusableRecomputeContext)
            if fail:
                raise RuntimeError("interrupted observation")
    except RuntimeError as error:
        assert fail and str(error) == "interrupted observation"
    assert Fuser.backward is original_backward
    assert rmsnorm.clear_tensor_data is original_clear
    assert final.checkpoint_fn.keywords["context_fn"] is original_context


def test_low_memory_checkpoint_mode_does_not_patch_transformer_engine(monkeypatch):
    def forbidden():
        raise AssertionError("Denoiser recomputation uses fresh backward contexts")

    monkeypatch.setattr("steerquant.map_reuse._load_transformer_engine", forbidden)
    with retained_backward_state(object(), enabled=False):
        pass


def test_unsupported_final_checkpoint_is_rejected_before_patching(monkeypatch):
    function = SimpleNamespace(backward=lambda ctx: None)
    original = function.backward
    rmsnorm = SimpleNamespace(clear_tensor_data=lambda value: None)
    monkeypatch.setattr("steerquant.map_reuse._load_transformer_engine", lambda: (function, rmsnorm))
    with pytest.raises(RuntimeError, match="selective checkpoint"):
        with retained_backward_state(SimpleNamespace(), enabled=True):
            pass
    assert function.backward is original


def test_real_selective_checkpoint_replays_cached_operations_for_retained_backward():
    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

    def contexts():
        counters = {}

        def policy(ctx, op, *args, **kwargs):
            key = (ctx.is_recompute, op)
            counters[key] = counters.get(key, 0) + 1
            return (
                CheckpointPolicy.MUST_SAVE
                if op == torch.ops.aten.mm.default
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        forward, recompute = create_selective_checkpoint_contexts(policy)
        return forward, ReusableRecomputeContext(recompute)

    torch.manual_seed(123)
    weight = torch.randn(4, 4)
    initial = torch.randn(2, 4)

    def function(x):
        return (x @ weight).sin() @ weight

    reused = initial.clone().requires_grad_(True)
    output = checkpoint(function, reused, use_reentrant=False, context_fn=contexts)
    for probe in range(16):
        baseline = initial.clone().requires_grad_(True)
        signs = torch.randn_like(output)
        (function(baseline) * signs).sum().backward()
        (output * signs).sum().backward(retain_graph=probe != 15)
        torch.testing.assert_close(reused.grad, baseline.grad, rtol=0, atol=0)
        reused.grad = None


def test_probe_interruption_clears_collector_captures():
    calls = []
    collector = SimpleNamespace(
        num_action_probes=16,
        reuse_forward=True,
        begin_pass=lambda: calls.append("clear"),
        validate_capture_count=lambda: None,
        set_probe=lambda index: None,
        validate_backward_count=lambda: None,
    )

    def interrupt(index, seconds):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_action_probes(
            lambda: torch.ones(2, requires_grad=True),
            collector,
            torch.Generator(),
            on_probe=interrupt,
        )
    assert calls == ["clear", "clear"]
