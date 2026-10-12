from hydra import initialize, compose
from omegaconf import OmegaConf
import pytest

TRAIN_CONFIGS = ["train_nucleus2_moe", "train_nucleus2_moe_divfree"]


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
def test_paths_and_directories_must_be_given(config_name):
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name)
    for key in ("data_dir", "log_dir"):
        assert OmegaConf.is_missing(cfg, key)
    for key in ("train_paths", "val_paths"):
        assert OmegaConf.is_missing(cfg.data_cfg, key)
    assert cfg.checkpoint_path is None
    assert cfg.smoke is False


@pytest.mark.parametrize("config_name", TRAIN_CONFIGS)
def test_model_config_names_a_registered_model(config_name):
    from nucleus.models import list_models

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name=config_name)
    assert cfg.model_cfg.name in list_models()
    assert cfg.model_cfg.name == config_name.removeprefix("train_")


def test_data_dir_interpolates_into_the_statistics_path():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="train_nucleus2_moe", overrides=["data_dir=/my/root"])
    assert cfg.data_cfg.statistics_path == "/my/root/statistics.json"
