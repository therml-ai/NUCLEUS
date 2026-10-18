from hydra import initialize, compose
from omegaconf import OmegaConf
import pytest

TRAIN_CONFIGS = ["train_nucleus2_moe", "train_nucleus2_moe_divfree_projection"]


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
def test_paths_and_directories_must_be_given(config_name):
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name)
    for key in ("split_dir", "log_dir"):
        assert OmegaConf.is_missing(cfg, key)
    assert cfg.checkpoint_path is None
    assert cfg.smoke is False


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
def test_model_config_names_a_registered_model(config_name):
    from nucleus.models import list_models

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name)
    assert cfg.model_cfg.name in list_models()
    assert cfg.model_cfg.name == config_name.removeprefix("train_")


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS + ["rollout"])
def test_the_statistics_live_in_the_split_directory(config_name):
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name, overrides=["split_dir=/my/split"])
    assert cfg.data_cfg.statistics_path == "/my/split/statistics.json"


def test_rollouts_take_the_runtime_options_of_a_model_config():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="rollout", overrides=["model_cfg.params.use_div_projection=true"])
    assert cfg.model_cfg.name == "nucleus2_moe_divfree_projection"
    assert cfg.model_cfg.params.use_div_projection is True
