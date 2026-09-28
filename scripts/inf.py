import os
import pathlib
import torch
import h5py
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from nucleus.models import load_model_from_checkpoint
from nucleus.data.normalize import get_normalizer
from nucleus.run_forward_trajectory import run_test, TestResults
from nucleus.trajectory import Trajectory
from nucleus.utils.set_fp32_precision import set_fp32_precision

def save_trajectory_as_hdf5(path, trajectory):
    with h5py.File(path, "w") as handle:
        if isinstance(trajectory, Trajectory):
            # Natural-grid fields: velocities on their staggered faces. Drop the
            # leading batch dim.
            fields = {
                "dfun": trajectory.sdf,
                "temperature": trajectory.temp,
                "velfacex": trajectory.velx,
                "velfacey": trajectory.vely,
            }
            for key, field in fields.items():
                handle.create_dataset(key, data=field.squeeze(0).cpu().detach().numpy())
        else:
            FIELDS = ["dfun", "temperature", "velx", "vely"]
            trajectory_np = trajectory.cpu().detach().numpy()
            for key, field in zip(FIELDS, np.split(trajectory_np, trajectory_np.shape[-1], -1)):
                handle.create_dataset(key, data=field)

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def main(cfg: DictConfig):
    set_fp32_precision()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass model_cfg so pre-"_extra_state" checkpoints (no embedded config) can still
    # be rebuilt; it is ignored for checkpoints that embed their own config.
    model = load_model_from_checkpoint(
        cfg.checkpoint_path,
        map_location=device,
        model_cfg=OmegaConf.to_container(cfg.model_cfg, resolve=True),
    )
    model = model.to(device)
    model.eval()

    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    
    # Rollouts are saved in the directory containing the checkpoint
    save_root = pathlib.Path(cfg.checkpoint_path).parent / "rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    
    with open(save_root / "config.yaml", "w") as handle:
        OmegaConf.save(cfg, f=handle.name)
    
    for test_file_path in cfg.data_cfg.test_paths:
        test_results: TestResults = run_test(cfg, model, normalizer, test_file_path, trajectory_steps=cfg.trajectory_steps)    
        rollout_save_root = save_root / test_results.case_name
        rollout_save_root.mkdir(parents=True, exist_ok=True)
        model_save_path = rollout_save_root / "model_cfg.json"
        with open(model_save_path, "w") as handle:
            OmegaConf.save(config=cfg.model_cfg, f=handle.name)
        save_trajectory_as_hdf5(rollout_save_root / "pred_trajectory.hdf5", test_results.preds)
        save_trajectory_as_hdf5(rollout_save_root / "gt_trajectory.hdf5", test_results.targets)
        json_save_path = rollout_save_root / "sim_params.json"
        with open(json_save_path, "w") as handle:
            json.dump(test_results.sim_params, handle)
        
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()