from copy import deepcopy
import pytest
import torch
from torch import nn

from steerquant.sensitivity import FirstOrderLinearCollector
from steerquant.integrations.linear_sites import LinearSite


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.aligned = nn.Linear(4, 4)
        self.context = nn.Linear(4, 4)
        self.register_buffer('text', torch.randn(1, 2, 4))

    def forward(self, x, sigma):
        return torch.tanh(self.aligned(x) + self.context(self.text).mean(1, keepdim=True)) / (1 + sigma)


@pytest.mark.parametrize('rotation_seed', [None, 1234])
def test_checkpoint_preserves_outputs_gradients_scores_and_releases_captures(monkeypatch, rotation_seed):
    def sites(net):
        return [LinearSite(1, 0, 'aligned', 'aligned', net.aligned, True),
                LinearSite(1, 1, 'context', 'context', net.context, False)]

    monkeypatch.setattr('steerquant.sensitivity.enumerate_block_linears', sites)
    torch.manual_seed(45)
    template = TinyNet().eval().requires_grad_(False)
    initial = torch.randn(1, 6, 4)

    def evaluate(checkpoint):
        net = deepcopy(template)
        collector = FirstOrderLinearCollector(net, weight_bits=4, activation_bits=8,
            activation_granularity='per_tensor',
            activation_clipping_ranges={'blocks.0.aligned': torch.ones(5) * 3,
                                        'blocks.0.context': torch.ones(5) * 3},
            num_action_probes=3, num_action_elements=24, num_denoising_steps=5,
            slot_names=('a', 'b', 'c'), checkpoint_denoisers=checkpoint,
            rotation_seed=rotation_seed)
        collector.register()
        gradients, outputs = [], []
        try:
            collector.reset_observation()
            for probe in range(3):
                collector.begin_pass()
                leaf = initial.clone().requires_grad_(True)
                x = leaf
                history = x * 0.1
                for call in range(5):
                    sigma = torch.tensor(0.1 * call)
                    collector.begin_denoise_call(call, sigma)
                    y = collector.checkpoint_call(net, x, sigma, call) if checkpoint else net(x, sigma)
                    x = x * 0.3 + y + history * 0.1
                    history = y
                collector.validate_capture_count()
                if checkpoint:
                    assert all(c.inputs is None and c.output is None for c in collector.captures)
                outputs.append(x.detach())
                collector.set_probe(probe)
                torch.manual_seed(100 + probe)
                (x * torch.randn_like(x)).sum().backward()
                collector.validate_backward_count()
                assert collector.tracker.total_calls == 5
                gradients.append(leaf.grad)
            rows = collector.export_action_probe_rows({'observation_id': 'test'})
            return outputs, gradients, rows
        finally:
            collector.remove()

    ordinary, checkpointed = evaluate(False), evaluate(True)
    for reference, actual in zip(ordinary[:2], checkpointed[:2]):
        for a, b in zip(reference, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ordinary[2] == checkpointed[2]
    assert len(checkpointed[2]) == 20
