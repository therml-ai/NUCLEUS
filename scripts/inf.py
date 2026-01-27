import os
import pathlib
import torch
from collections import OrderedDict
from bubbleformer.models import get_model
import hydra
from omegaconf import DictConfig
from bubbleformer.test import run_test, TestResults
from bubbleformer.plot.plotting import plot_rollout
from bubbleformer.utils.set_fp32_precision import set_fp32_precision

@hydra.main(version_base=None, config_path="../bubbleformer/config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()
    
    model_name = cfg.model_cfg.name
    model_kwargs = {
        "input_fields": 4,
        "output_fields": 4,
        "time_window": cfg.data_cfg.time_window,
        "patch_size": cfg.model_cfg.params.patch_size,
        "embed_dim": cfg.model_cfg.params.embed_dim,
        "processor_blocks": cfg.model_cfg.params.processor_blocks,
        "num_heads": cfg.model_cfg.params.num_heads,
        "num_experts": cfg.model_cfg.params.num_experts,
        "topk": cfg.model_cfg.params.topk,
        "load_balance_loss_weight": cfg.model_cfg.params.load_balance_loss_weight,
        "num_fluid_params": cfg.model_cfg.params.num_fluid_params,
    }

    model = get_model(model_name, **model_kwargs)
    model = model.cuda()
    model_data = torch.load(cfg.checkpoint_path, weights_only=False)
    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        name = key[6:]
        weight_state_dict[name] = val
    del model_data
    model.load_state_dict(weight_state_dict)
    model.eval()
    
    # Rollouts are saved in the directory containing the checkpoint
    save_root = pathlib.Path(cfg.checkpoint_path).parent / "inference_rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    for test_file_path in cfg.data_cfg.test_paths:
        test_results: TestResults = run_test(model, test_file_path, max_timesteps=1000)
        setup = test_results.fluid_params["setup"]
        liquid = test_results.fluid_params["liquid"]
        heater_temp = test_results.fluid_params["heater"]["wallTemp"]
        
        save_dir = save_root / f"{setup}_{liquid}_{heater_temp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_rollout(save_dir, test_results.preds, test_results, step_size=20)
        
        torch.save(test_results, save_dir / "test_results.pt")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
