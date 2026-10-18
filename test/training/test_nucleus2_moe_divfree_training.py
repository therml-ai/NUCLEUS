import math

import pytest
import torch

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFree
from nucleus.physics.coordinates import CellCenteredGrid
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.testing.flashx import flashx_normalizer, random_batch, random_forecast_batch
from nucleus.training.modules.nucleus2_moe_divfree import (
    Nucleus2MoEDivFreeTraining,
    Nucleus2MoEDivFreeTrainingSettings,
    target_div_source,
)

FIELD_NAMES = Nucleus2MoEDivFree.field_names


@pytest.fixture
def divfree_training():
    torch.manual_seed(0)
    normalizer = flashx_normalizer(FIELD_NAMES, vector_fields=Nucleus2MoEDivFree.vector_fields)
    model = get_model(
        "nucleus2_moe_divfree", normalizer, patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )
    return Nucleus2MoEDivFreeTraining(model, Nucleus2MoEDivFreeTrainingSettings(target_timesteps=2))


def _forecast_batch(target_timesteps=2):
    return random_forecast_batch(FIELD_NAMES, Nucleus2MoEDivFree.config_keys, target_timesteps=target_timesteps)


def test_training_step_gives_a_finite_loss_that_backpropagates(divfree_training):
    output = divfree_training.training_step(_forecast_batch(), step=0)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradient_metrics = divfree_training.gradient_metrics()
    assert set(gradient_metrics) == {
        "train/debed/grad_norm", "train/debed/grad_absmax", "train/debed/weight_std", "train/debed/weight_absmax",
    }
    assert all(torch.isfinite(value) for value in gradient_metrics.values())
    for name in ("dfun", "velfacex", "div_source"):
        assert f"train/mae_{name}" in output.metrics


def test_validation_step_scores_the_first_target_window(divfree_training):
    divfree_training.eval()
    with torch.no_grad():
        metrics = divfree_training.validation_step(_forecast_batch(target_timesteps=4))
    assert math.isfinite(float(metrics["val/loss"]))


def test_push_forward_trains_on_a_later_target_window(divfree_training):
    training = Nucleus2MoEDivFreeTraining(
        divfree_training.model,
        Nucleus2MoEDivFreeTrainingSettings(target_timesteps=2, max_unrolls=1, push_forward_probability=1.0),
    )
    output = training.training_step(_forecast_batch(target_timesteps=4), step=0)
    assert output.metrics["train/num_unrolls"] == 1
    assert torch.isfinite(output.loss)


def test_the_divergence_source_is_supervised_by_the_target_divergence():
    target = random_batch(FIELD_NAMES, Nucleus2MoEDivFree.config_keys, batch_size=1)
    grid = CellCenteredGrid(1 / 16, 1 / 16, torch.zeros(1, 1, 1, 32))
    velx, vely = target.fields["velfacex"], target.fields["velfacey"]
    torch.testing.assert_close(
        target_div_source(target, grid), divergence_centers_from_faces(velx, vely, grid.dx, grid.dy)
    )
