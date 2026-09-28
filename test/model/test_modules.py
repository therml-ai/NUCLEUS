from hydra import initialize, compose
import torch

from nucleus.data.batching import CollatedBatch
from nucleus.models.modules import get_train_module

def test_phase_forecast_module():
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="default",
            overrides=["model_cfg=nucleus2/nucleus2_phase", "data_dir=/tmp"],
        )
        
    Module = get_train_module(cfg.model_cfg.train_module_name)
    module = Module(
        None,
        cfg.model_cfg,
        cfg.data_cfg,
        cfg.normalizer_cfg,
        cfg.optim_cfg,
        cfg.scheduler_cfg,
        log_wandb=False,
        normalization_constants=None
    )
    module.default_log_dict = lambda *args, **kwargs: None # force no logging
    
    device = "cpu"
    batch = CollatedBatch(
        input=torch.randn(2, 2, 64, 64, 4, device=device),
        target=torch.randn(2, 2, 64, 64, 4, device=device),
        sim_params_dict={},
        sim_params_tensor=torch.randn(2, module.model.num_sim_params, device=device),
        x_grid=torch.randn(64, device=device),
        y_grid=torch.randn(64, device=device),
        dx=torch.tensor(0.01, device=device),
        dy=torch.tensor(0.01, device=device),
    )

    @torch.compiler.disable
    def check_step():
        with torch.no_grad():
            loss = module.validation_step(batch, 0)
            assert torch.isfinite(loss)
            assert loss > 0
    check_step()