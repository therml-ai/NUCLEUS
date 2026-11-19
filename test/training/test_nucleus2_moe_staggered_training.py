import math

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_staggered import Nucleus2MoEStaggered
from nucleus.testing.flashx import flashx_normalizer, random_forecast_batch
from nucleus.training.modules.nucleus2_moe_staggered import (
    Nucleus2MoEStaggeredTraining,
    Nucleus2MoEStaggeredTrainingSettings,
)

FIELD_NAMES = Nucleus2MoEStaggered.field_names


@pytest.fixture
def staggered_training():
    torch.manual_seed(0)
    normalizer = flashx_normalizer(FIELD_NAMES)
    model = get_model(
        "nucleus2_moe_staggered", normalizer, patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )
    return Nucleus2MoEStaggeredTraining(model, Nucleus2MoEStaggeredTrainingSettings(target_timesteps=2))


def _forecast_batch(target_timesteps=2):
    return random_forecast_batch(FIELD_NAMES, Nucleus2MoEStaggered.config_keys, target_timesteps=target_timesteps)


def test_training_step_gives_a_finite_loss_that_backpropagates(staggered_training):
    output = staggered_training.training_step(_forecast_batch(), step=0)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    for name in FIELD_NAMES:
        assert f"train/mae_{name}" in output.metrics
    # the routers are balanced by a bias, so the loss is the data loss alone
    torch.testing.assert_close(output.loss, output.metrics["train/data_loss"])


def test_validation_step_scores_the_first_target_window(staggered_training):
    staggered_training.eval()
    with torch.no_grad():
        metrics = staggered_training.validation_step(_forecast_batch(target_timesteps=4))
    assert math.isfinite(float(metrics["val/loss"]))


def test_push_forward_trains_on_a_later_target_window(staggered_training):
    training = Nucleus2MoEStaggeredTraining(
        staggered_training.model,
        Nucleus2MoEStaggeredTrainingSettings(target_timesteps=2, max_unrolls=1, push_forward_probability=1.0),
    )
    output = training.training_step(_forecast_batch(target_timesteps=4), step=0)
    assert output.metrics["train/num_unrolls"] == 1
    assert torch.isfinite(output.loss)


def test_the_train_config_builds_the_training_settings():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="train_nucleus2_moe_staggered")
    Nucleus2MoEStaggeredTrainingSettings(target_timesteps=cfg.data.target_timesteps, **OmegaConf.to_container(cfg.training))
