import glob
import subprocess
import pathlib

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    model = "neighbor_vit_poolboiling64"
    job_id = "48103654"
    
    rollout_dir = pathlib.Path(f"{cfg.log_dir}/{model}_{job_id}/checkpoints/inference_rollouts/")
    
    for rollout_path in rollout_dir.glob("**/*/"):
        problem = rollout_path.name
        print(problem)
        # NOTE: this assumes imagemagick is available on the system
        result = subprocess.run(f"convert {rollout_path}/rollout_*.png {rollout_path}/rollout.gif", shell=True) 
        if result.returncode != 0:
            print(f"Error converting rollout for {problem}")
            print(result.stderr)
            continue
        print(f"Converted rollout for {problem}")

if __name__ == "__main__":
    main()