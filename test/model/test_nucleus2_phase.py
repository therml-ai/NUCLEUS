import pytest
import torch

from nucleus.models.nucleus2_moe import Nucleus2MoEConfig
from nucleus.models.nucleus2_moe_phase import Nucleus2MoEPhase


@pytest.fixture
def config():
    return Nucleus2MoEConfig(
        patch_size=4,
        embed_dim=64,
        num_heads=2,
        processor_blocks=2,
        num_experts=4,
        topk=2,
        moe_intermediate_dim=64,
        patching="Linear",
    )


@pytest.fixture
def model(config):
    return Nucleus2MoEPhase(config)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_nucleus2_phase_step(device, model, batch_size, config):
    #if device == "cpu":
    #    pytest.skip("flex attention not supported on CPU")

    model = model.to(device)
    B, T, H, W = batch_size, 2, 32, 32

    fields = torch.randn(B, T, H, W, 3, device=device)
    phase = torch.randint(0, 2, (B, T, H, W), dtype=torch.int32, device=device)
    sim_params = torch.randn(B, model.num_sim_params, device=device)

    with torch.no_grad():
        pred_fields, pred_phase_logits, moe_outputs = model.step(fields, phase, sim_params)

    assert pred_fields.shape == (B, T, H, W, 3)
    assert pred_phase_logits.shape == (B, T, H, W)
    assert torch.all(torch.isfinite(pred_fields))
    assert torch.all(torch.isfinite(pred_phase_logits))


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("trajectory_steps", [4, 8])
@pytest.mark.parametrize("return_moe_outputs", [True, False])
def test_nucleus2_phase_forward_trajectory(device, model, batch_size, trajectory_steps, return_moe_outputs):
    model = model.to(device)
    B, T, H, W = batch_size, 2, 32, 32

    # initial_state is 4-channel: phase (channel 0, binary 0/1) + fields (channels 1-3)
    phase = torch.randint(0, 2, (B, T, H, W, 1), dtype=torch.float32, device=device)
    fields = torch.randn(B, T, H, W, 3, device=device)
    initial_state = torch.cat([phase, fields], dim=-1)
    sim_params = torch.randn(B, model.num_sim_params, device=device)

    with torch.no_grad():
        result = model.forward_trajectory(
            initial_state=initial_state,
            sim_params=sim_params,
            dx=0.01,
            input_time_window_size=T,
            output_time_window_size=T,
            trajectory_steps=trajectory_steps,
            return_moe_outputs=return_moe_outputs,
        )

    trajectory = result[0] if return_moe_outputs else result
    assert trajectory.shape == (B, trajectory_steps, H, W, 4)
    assert torch.all(torch.isfinite(trajectory))
