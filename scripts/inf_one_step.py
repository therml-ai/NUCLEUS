import os
import pathlib
import torch
from collections import OrderedDict
from nucleus.models import get_model
import hydra
from omegaconf import DictConfig
from nucleus.test_one_step import run_test_one_step
from nucleus.utils.set_fp32_precision import set_fp32_precision

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()

    print(f"Processing checkpoint: {cfg.checkpoint_path}")    
    
    model_name = cfg.model_cfg.name
    model_kwargs = {
        "input_fields": 4,
        "output_fields": 4,
        "patch_size": cfg.model_cfg.params.patch_size,
        "embed_dim": cfg.model_cfg.params.embed_dim,
        "processor_blocks": cfg.model_cfg.params.processor_blocks,
        "num_heads": cfg.model_cfg.params.num_heads,
        "num_fluid_params": cfg.model_cfg.params.num_fluid_params,
    }
    
    if cfg.model_cfg.params.get("num_experts", None) is not None:
        model_kwargs["num_experts"] = cfg.model_cfg.params.num_experts
        model_kwargs["topk"] = cfg.model_cfg.params.topk
        model_kwargs["load_balance_loss_weight"] = cfg.model_cfg.params.load_balance_loss_weight

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
    
    all_test_results = run_test_one_step(model, cfg)
    save_dir = pathlib.Path(cfg.checkpoint_path).parent / "inference_rollouts"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(all_test_results, save_dir / "test_one_step.pt")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
