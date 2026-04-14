import os
import pathlib
import torch
from collections import OrderedDict
from nucleus.models import get_model
import hydra
from omegaconf import DictConfig, OmegaConf
from nucleus.data.normalize import get_normalizer
from nucleus.test import run_test, TestResults
from nucleus.plot.plotting import (
    plot_rollout, 
    plot_rollout_stability, 
    plot_rollout_moe_overlay,
)
from nucleus.plot.plot_metrics import (
    plot_simple_metrics,
    plot_vapor_volume_at_height,
    plot_bubble_counts,
)
from nucleus.utils.set_fp32_precision import set_fp32_precision
from lightning import LightningModule

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = cfg.model_cfg.name
    #model_kwargs = {
    #    "input_fields": 4,
    #    "output_fields": 4,
    #    "patch_size": cfg.model_cfg.params.patch_size,
    #    "embed_dim": cfg.model_cfg.params.embed_dim,
    #    "processor_blocks": cfg.model_cfg.params.processor_blocks,
    #    "num_heads": cfg.model_cfg.params.num_heads,
    #    "num_fluid_params": cfg.model_cfg.params.num_fluid_params,
    #}
    
    model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)
    
    if cfg.model_cfg.params.get("num_experts", None) is not None:
        model_kwargs["num_experts"] = cfg.model_cfg.params.num_experts
        model_kwargs["topk"] = cfg.model_cfg.params.topk
    
    if "load_balance_loss_weight" in cfg.model_cfg.params:
        model_kwargs["load_balance_loss_weight"] = cfg.model_cfg.params.load_balance_loss_weight
        
    model = get_model(model_name, **model_kwargs)
    model = model.to(device)
    model_data = torch.load(cfg.checkpoint_path, weights_only=False, map_location=device)
    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        print(key, val.shape)
        if isinstance(model, LightningModule):
            name = key
        else:
            name = key[6:]
        weight_state_dict[name] = val
    del model_data
    model.load_state_dict(weight_state_dict)
    model.eval()
    
    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    
    # Rollouts are saved in the directory containing the checkpoint
    save_root = pathlib.Path(cfg.checkpoint_path).parent / "inference_rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    all_test_results = []
    for test_file_path in cfg.data_cfg.test_paths:
        test_results: TestResults = run_test(cfg, model, normalizer, test_file_path, max_timesteps=300)
        all_test_results.append(test_results)

        save_dir = save_root / f"{test_results.case_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_rollout(
            save_dir=save_dir,
            rollout=test_results.preds,
            test_results=test_results,
            step_size=5,
        )
        
    torch.save(all_test_results, save_root / "test_results_reinit.pt")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
