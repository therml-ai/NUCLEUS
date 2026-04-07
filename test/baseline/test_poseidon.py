import torch

from nucleus.baseline.poseidon import ScOT, ScOTConfig, ScOTOutput

def test_poseidon():
    cfg = ScOTConfig(
        image_size=64,
        patch_size=4,
        num_channels=4,
        num_out_channels=4,
    )
    
    model = ScOT(cfg)
    input = torch.randn(4, 4, 64, 64)
    time = torch.randn(4)
    output: ScOTOutput = model(input, time)
    
    assert torch.isfinite(output.output).all()
    
    loss = output.output.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()