from hydra import initialize, compose
import os

def test_default_log_dir():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="default")
        assert cfg.log_dir == os.environ["TMPDIR"]
        
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