import torch
from nucleus.baseline.moe_dpot import MoEPOTNet
from nucleus.data.batching import CollatedBatch
from nucleus.testing.parametrize import parametrize_available_devices

@parametrize_available_devices("device")
def test_moe_dpot(device):
    
    config = {
        "in_channels": 4,
        "out_channels": 4,
        "in_timesteps": 8,
        "out_timesteps": 8,
        "n_blocks": 8,
        "modes": 32,
        "embed_dim": 768,
        "mlp_ratio": 1,
        "act": "gelu",
        "img_size": 64,
        "patch_size": 16,
        "normalize": True,
        "time_agg": "mlp",
        "n_cls": 6,
        "is_finetune": False,
        "mixing_type": "afno",
        "depth": 8,
        "out_layer_dim": 256,
    }

    model = MoEPOTNet(config).to(device)
    
    input = torch.randn(4, 64, 64, 8, 4).to(device).requires_grad_(True)
    output, cls_pred, router_loss_total = model(input)
    assert output.shape == (4, 64, 64, 8, 4)
    assert torch.all(torch.isfinite(output))
    assert cls_pred.shape == (4, 6)
    assert torch.all(torch.isfinite(cls_pred))
    assert router_loss_total.shape == ()
    assert torch.all(torch.isfinite(router_loss_total))
    assert router_loss_total.item() > 0 # extremely unlikely to perfectly route
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = cls_pred.sum() + cls_pred.sum() + router_loss_total
    loss.backward()
    
    # check that gradients are finite
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad))
    
    # check that parameters after optimizer are finite.
    optimizer.step()
    for param in model.parameters():
        assert torch.all(torch.isfinite(param.data))        