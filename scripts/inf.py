import os
import pathlib
import torch
import h5py
import json
from collections import OrderedDict
from nucleus.models import get_model
import hydra
from omegaconf import DictConfig, OmegaConf
from nucleus.data.normalize import get_normalizer
from nucleus.run_forward_trajectory import run_test, TestResults
from nucleus.plot.plotting import (
    plot_rollout,
    plot_rollout_stability,
    plot_rollout_moe_overlay,
    plot_distribution,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model_cfg.name
    model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)
    model = get_model(model_name, **model_kwargs)
    model = model.to(device)
    
    model_data = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)    
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
    save_root = pathlib.Path(cfg.checkpoint_path).parent / "rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    all_test_results = []
    for test_file_path in cfg.data_cfg.test_paths:

        test_results: TestResults = run_test(cfg, model, normalizer, test_file_path, trajectory_steps=300)
        all_test_results.append(test_results)

        save_dir = save_root / f"{test_results.case_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_rollout(
           save_dir=save_dir,
           rollout=test_results.preds,
           test_results=test_results,
           step_size=5,
            include_ground_truth=True,
        )
        plot_distribution(
            save_dir=save_dir,
            rollout=test_results.preds,
            test_results=test_results,
        )
     
    for test_result in all_test_results:
        h5py_save_path = save_root / f"{test_result.case_name}.hdf5"
        with h5py.File(h5py_save_path, "w") as handle:
            handle.create_dataset("pred_trajectory", data=test_result.preds)
            handle.create_dataset("gt_trajectory", data=test_result.targets)
        
        
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()