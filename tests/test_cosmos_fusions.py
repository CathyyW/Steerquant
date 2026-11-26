import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cosmos RMSNorm requires CUDA")
@pytest.mark.parametrize("fp32_strategy", [False, True])
def test_fusion_adapter_preserves_upstream_block(fp32_strategy):
    pytest.importorskip("cosmos_policy._src.predict2.networks.minimal_v4_dit")
    from cosmos_policy._src.predict2.networks.minimal_v4_dit import Block
    from steerquant.integrations.fusions import install_fusions

    torch.manual_seed(195)
    block = Block(64, 64, 1, backend="torch", use_wan_fp32_strategy=fp32_strategy)
    block = block.cuda().to(torch.bfloat16).eval()
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for shape in ((1, 3, 2, 2, 64), (1, 3, 64), (1, 8, 64))
    ]
    with torch.no_grad():
        expected = block(*inputs)
        install_fusions(block)
        actual = block(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
