import torch
import pytest

from nucleus.models import get_model
from nucleus.data.batching import CollatedBatch


@pytest.fixture
def model():
    return get_model(
        "nucleus2_moe",
        input_fields=4,
        output_fields=4,
        patch_size=4,
        embed_dim=128,
        num_heads=2,
        processor_blocks=2,
        num_experts=4,
        topk=2,
        mlp_ratio=4.0
    )

@pytest.mark.parametrize("device", ["cpu"])
def test_nucleus2(device, model):
    model = model.to(device)
    
    batch = CollatedBatch(
        input=torch.randn(4, 8, 64, 64, 4, device=device),
        target=None,
        sim_params_dict={},
        sim_params_tensor=torch.randn(4, model.num_sim_params, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )
    output, moe_output = model(batch)
    assert output.shape == (4, 8, 64, 64, 4)
    assert torch.all(torch.isfinite(output))

    loss = output.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad))
            
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("trajectory_steps", [8, 24, 32])
@pytest.mark.parametrize("use_sdf_reinit", [True, False])
@pytest.mark.parametrize("return_moe_outputs", [True, False])
def test_nucleus2_forward_trajectory(
    device,
    model, 
    batch_size,
    trajectory_steps,
    use_sdf_reinit,
    return_moe_outputs
):
    model = model.to(device)
    batch = CollatedBatch(
        input=torch.randn(batch_size, 8, 64, 64, 4, device=device),
        target=None,
        sim_params_dict={},
        sim_params_tensor=torch.randn(batch_size, model.num_sim_params, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )
    
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
    assert trajectory.shape[-1] == 4