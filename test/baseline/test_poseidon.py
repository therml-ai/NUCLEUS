import torch
import pytest

from nucleus.data.batching import CollatedBatch
from nucleus.testing.parametrize import parametrize_available_devices

try:
    import transformers
    HAS_TRANFORMERS_DEPS = True
except ImportError:
    HAS_TRANFORMERS_DEPS = False
    
if HAS_TRANFORMERS_DEPS:
    from nucleus.baseline.poseidon import ScOT, ScOTConfig, ScOTOutput

@pytest.mark.skipif(not HAS_TRANFORMERS_DEPS, reason="poseidon dependencies not installed")
@parametrize_available_devices("device")
def test_poseidon(device):
    cfg = ScOTConfig(
        image_size=64,
        patch_size=4,
        num_channels=4,
        num_out_channels=4,
    )
    
    model = ScOT(cfg).to(device)
    
    batch = CollatedBatch(
        input=torch.randn(4, 4, 64, 64).to(device),
        target=torch.randn(4, 4, 64, 64).to(device),
        fluid_params_dict={},
        fluid_params_tensor=torch.randn(4, 16, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )
    
    time = torch.randn(4).to(device)
    output: ScOTOutput = model(batch)
    
    assert torch.isfinite(output.output).all()
    
    loss = output.output.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()