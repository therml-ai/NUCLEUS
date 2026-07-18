import random
import torch
import pytest

from nucleus.models.nucleus2_moe import Nucleus2MoE, Nucleus2MoEConfig
from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFree, Nucleus2MoEDivFreeConfig
from nucleus.data.batching import CollatedBatch
from nucleus.data.normalize import NoNormalizer


# Exercise both the unconstrained model and the divergence-free variant. They
# share the MoEBase interface (forward(batch) -> step(input, sim_params)); the
# div-free model only differs in how it reconstructs the velocity fields.
MODELS = [
    (Nucleus2MoE, Nucleus2MoEConfig),
    (Nucleus2MoEDivFree, Nucleus2MoEDivFreeConfig),
]


@pytest.fixture(params=MODELS, ids=["unconstrained", "divfree"])
def model(request):
    model_cls, config_cls = request.param
    return model_cls(config_cls(
        patch_size=4,
        embed_dim=64,
        num_heads=2,
        processor_blocks=2,
        num_experts=4,
        topk=2,
        moe_intermediate_dim=64,
        patching="Linear",
    ))


def make_batch(batch_size, num_sim_params, device):
    return CollatedBatch(
        input=torch.randn(batch_size, 8, 64, 64, 4, device=device),
        target=None,
        sim_params_dict={},
        sim_params_tensor=torch.randn(batch_size, num_sim_params, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )


def make_sim_params_dict(model):
    # forward_trajectory assembles its conditioning tensor from a physical sim-param
    # dict; the values are arbitrary for a finiteness/shape check.
    params = {"bulk_temp": 50.0, "sat_temp": 58.0}
    params.update({param: random.random() for param in model.expected_fluid_params})
    params["heater"] = {param: random.random() for param in model.expected_heater_params}
    params.update({param: random.random() for param in model.expected_global_params})
    return params


@pytest.mark.parametrize("device", ["cpu"])
def test_nucleus2(device, model):
    if device == "cpu":
        pytest.skip("flex attention not supported on CPU")

    model = model.to(device)
    batch = make_batch(2, model.num_sim_params, device)

    output, moe_output = model(batch)
    assert output.shape == (2, 8, 64, 64, 4)
    assert torch.all(torch.isfinite(output))

    loss = output.sum()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(torch.isfinite(param.grad))


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("trajectory_steps", [8, 24])
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
    batch = make_batch(batch_size, model.num_sim_params, device)

    with torch.inference_mode():
        # forward_trajectory runs in physical space and normalizes internally; an
        # identity normalizer keeps the random-input finiteness/shape checks valid.
        trajectory = model.forward_trajectory(
            batch.input,
            make_sim_params_dict(model),
            NoNormalizer(),
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
