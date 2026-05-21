from hydra import initialize, compose
import os

def test_layout():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default", overrides=["model_cfg=nucleus2/nucleus2_experiment"])
        assert cfg.log_dir == os.environ["TMPDIR"]
        assert cfg.model_cfg.layout == "t h w c"