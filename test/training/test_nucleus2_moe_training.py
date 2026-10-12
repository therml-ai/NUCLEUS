import math

import torch

from nucleus.models.nucleus2_moe import Nucleus2MoE
from nucleus.testing.flashx import random_forecast_batch
from nucleus.training.modules.nucleus2_moe import Nucleus2MoETraining, Nucleus2MoETrainingSettings


def _forecast_batch(target_timesteps=2):
    return random_forecast_batch(Nucleus2MoE.field_names, Nucleus2MoE.config_keys, target_timesteps=target_timesteps)


def test_training_step_gives_a_finite_loss_that_backpropagates(moe_training):
    output = moe_training.training_step(_forecast_batch(), step=0)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    grads = [param.grad for param in moe_training.model.parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert "train/data_loss" in output.metrics


def test_validation_step_scores_the_first_target_window(moe_training):
    moe_training.eval()
    with torch.no_grad():
        metrics = moe_training.validation_step(_forecast_batch(target_timesteps=4))
    assert math.isfinite(float(metrics["val/loss"]))


def test_push_forward_trains_on_a_later_target_window(moe_training):
    training = Nucleus2MoETraining(
        moe_training.model,
        Nucleus2MoETrainingSettings(target_timesteps=2, max_unrolls=1, push_forward_probability=1.0),
    )
    output = training.training_step(_forecast_batch(target_timesteps=4), step=0)
    assert output.metrics["train/num_unrolls"] == 1
    assert torch.isfinite(output.loss)
