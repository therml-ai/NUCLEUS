import torch
import pytest

from nucleus.baseline.poseidon import ScOT, ScOTConfig, ScOTOutput
from nucleus.testing.parametrize import parametrize_available_devices

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
@parametrize_available_devices("device")
def test_poseidon(device):
    cfg = ScOTConfig(
        image_size=64,
        patch_size=4,
        num_channels=4,
        num_out_channels=4,
    )
    
    model = ScOT(cfg).to(device)
    input = torch.randn(4, 4, 64, 64).to(device)
    time = torch.randn(4).to(device)
    output: ScOTOutput = model(input, time)
    
    assert torch.isfinite(output.output).all()
    
    loss = output.output.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()