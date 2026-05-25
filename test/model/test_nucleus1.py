import torch
import pytest

from nucleus.models import get_model
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

def _get_model(model_name):
    kwargs = dict(
        input_fields=4,
        output_fields=4,
        patch_size=4,
        embed_dim=128,
        num_heads=2,
        processor_blocks=2,
    )
    if "moe" in model_name:
        kwargs["num_experts"] = 4
        kwargs["topk"] = 2
    return get_model(model_name, **kwargs)

@pytest.mark.parametrize("model_name", _MOE_MODELS)
@pytest.mark.parametrize("device", ["cpu"])
def test_nucleus1_moe(device, model_name):
    
    model = _get_model(model_name)
    model = model.to(device)
    
    batch = CollatedBatch(
        input=torch.randn(4, 8, 4, 64, 64, device=device),
        target=None,
        sim_params_dict={},
        sim_params_tensor=torch.randn(4, model.num_sim_params, device=device),
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
            
@pytest.mark.parametrize("model_name", _VIT_MODELS)
@pytest.mark.parametrize("device", ["cpu"])
def test_nucleus1_vit(device, model_name):
    
    model = _get_model(model_name)
    model = model.to(device)
    
    batch = CollatedBatch(
        input=torch.randn(4, 8, 4, 64, 64, device=device),
        target=None,
        sim_params_dict={},
        sim_params_tensor=torch.randn(4, model.num_sim_params, device=device),
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
            
@pytest.mark.parametrize("model_name", _VIT_MODELS + _MOE_MODELS)
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("trajectory_steps", [8, 24, 32])
@pytest.mark.parametrize("use_sdf_reinit", [True, False])
def test_nucleus1_forward_trajectory(
    device,
    model_name, 
    batch_size,
    trajectory_steps,
    use_sdf_reinit,
):
    model = _get_model(model_name)
    model = model.to(device)   
    batch = CollatedBatch(
        input=torch.randn(batch_size, 8, 4, 64, 64, device=device),
        target=None,
        sim_params_dict={},
        sim_params_tensor=torch.randn(batch_size, model.num_sim_params, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )
    
    return_moe_outputs = "_moe" in model_name
    trajectory = model.forward_trajectory(
        initial_state=batch.input,
        sim_params=batch.sim_params_tensor,
        dx=1/4,
        input_time_window_size=8,
        output_time_window_size=8,
        trajectory_steps=trajectory_steps,
        use_sdf_reinit=use_sdf_reinit,
        return_moe_outputs=return_moe_outputs
    )
    if return_moe_outputs:
        trajectory, moe_outputs = trajectory
    
    assert trajectory.isfinite().all()
    assert trajectory.shape[0] == batch_size
    assert trajectory.shape[1] == trajectory_steps
    assert trajectory.shape[2] == 4