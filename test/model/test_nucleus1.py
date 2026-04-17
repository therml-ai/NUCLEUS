import torch
import pytest

from nucleus.models import get_model
from nucleus.testing.parametrize import parametrize_available_devices
from nucleus.data.batching import CollatedBatch

_MOE_MODELS = [
    "nucleus1_vit_moe",
    "nucleus1_axial_moe",
    "nucleus1_moe",
]

_VIT_MODELS = [
    "nucleus1_vit",
    "nucleus1_axial_vit",
    "nucleus1_neighbor_vit",
]

@parametrize_available_devices("device")
@pytest.mark.parametrize("model_name", _MOE_MODELS)
def test_nucleus1_moe(device, model_name):
    
    model = get_model(model_name,
        input_fields=4,
        output_fields=4,
        patch_size=4,
        embed_dim=128,
        num_heads=2,
        processor_blocks=4,
        num_fluid_params=16,
        num_experts=4,
        topk=2,
    )
    model = model.to(device)
    
    batch = CollatedBatch(
        input=torch.randn(4, 8, 4, 64, 64, device=device),
        target=None,
        fluid_params_dict={},
        fluid_params_tensor=torch.randn(4, 16, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )
    
    output, moe_output = model(batch)
    assert output.shape == (4, 8, 4, 64, 64)
    assert torch.all(torch.isfinite(output))
    
    moe_loss = sum([m.load_balance_loss for m in moe_output])
    loss = output.sum() + moe_loss
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad))
            
@parametrize_available_devices("device")
@pytest.mark.parametrize("model_name", _VIT_MODELS)
def test_nucleus1_vit(device, model_name):
    
    model = get_model(model_name,
        input_fields=4,
        output_fields=4,
        patch_size=4,
        embed_dim=128,
        num_heads=2,
        processor_blocks=2,
        num_fluid_params=16,
    )
    model = model.to(device)
    
    batch = CollatedBatch(
        input=torch.randn(4, 8, 4, 64, 64, device=device),
        target=None,
        fluid_params_dict={},
        fluid_params_tensor=torch.randn(4, 16, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )
    
    output = model(batch)
    assert output.shape == (4, 8, 4, 64, 64)
    assert torch.all(torch.isfinite(output))
    
    loss = output.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad))