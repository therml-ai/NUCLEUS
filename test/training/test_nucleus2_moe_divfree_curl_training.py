import math

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_divfree_curl import Nucleus2MoEDivFreeCurl
from nucleus.testing.flashx import flashx_normalizer, random_forecast_batch
from nucleus.training.modules.nucleus2_moe_divfree_curl import (
    Nucleus2MoEDivFreeCurlTraining,
    Nucleus2MoEDivFreeCurlTrainingSettings,
)

FIELD_NAMES = Nucleus2MoEDivFreeCurl.field_names


def _training(**settings) -> Nucleus2MoEDivFreeCurlTraining:
    torch.manual_seed(0)
    normalizer = flashx_normalizer(FIELD_NAMES, vector_fields=Nucleus2MoEDivFreeCurl.vector_fields)
    model = get_model(
        "nucleus2_moe_divfree_curl", normalizer, patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )
    return Nucleus2MoEDivFreeCurlTraining(model, Nucleus2MoEDivFreeCurlTrainingSettings(target_timesteps=2, **settings))


def _forecast_batch(target_timesteps=2):
    return random_forecast_batch(FIELD_NAMES, Nucleus2MoEDivFreeCurl.config_keys, target_timesteps=target_timesteps)


def test_training_step_gives_a_finite_loss_that_backpropagates():
    training = _training()
    output = training.training_step(_forecast_batch(), step=0)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(torch.isfinite(value) for value in training.gradient_metrics().values())
    for name in (*FIELD_NAMES, "div_source"):
        assert f"train/mae_{name}" in output.metrics


def test_validation_step_scores_the_first_target_window():
    training = _training().eval()
    with torch.no_grad():
        metrics = training.validation_step(_forecast_batch(target_timesteps=4))
    assert math.isfinite(float(metrics["val/loss"]))


def test_push_forward_trains_on_a_later_target_window():
    training = _training(max_unrolls=1, push_forward_probability=1.0)
    output = training.training_step(_forecast_batch(target_timesteps=4), step=0)
    assert output.metrics["train/num_unrolls"] == 1
    assert torch.isfinite(output.loss)


def test_the_train_config_builds_the_training_settings():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="train_nucleus2_moe_divfree_curl")
    assert cfg.model_cfg.name == "nucleus2_moe_divfree_curl"
    Nucleus2MoEDivFreeCurlTrainingSettings(target_timesteps=cfg.data.target_timesteps, **OmegaConf.to_container(cfg.training))
