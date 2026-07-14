from hydra import initialize, compose
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
import pytest


def test_default_mandatory_fields_missing():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
        assert OmegaConf.is_missing(cfg, "data_dir")
        assert OmegaConf.is_missing(cfg, "log_dir")
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.log_dir


def test_default_empty_fields():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
        assert cfg.checkpoint_path is None
        assert cfg.commit_sha is None


def test_required_fields():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
        assert cfg.batch_size is not None
        assert cfg.accumulate_grad_batches is not None
        assert cfg.history_time_window is not None
        assert cfg.future_time_window is not None
        assert cfg.time_step is not None
        assert cfg.start_time is not None


def test_data_dir_interpolates_into_paths():
    # Providing data_dir on the CLI resolves the ${data_dir} interpolation used as
    # the root of every path in the data_cfg.
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default", overrides=["data_dir=/my/root"])
        first_path = cfg.data_cfg.train_paths[0]
        assert first_path.startswith("/my/root/")
        assert "${data_dir}" not in first_path
