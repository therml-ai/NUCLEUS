import glob
import subprocess
import pathlib

#model = "neighbor_moe_poolboiling64"
model = "neighbor_vit_poolboiling64"
job_id = "48103654"

rollout_dir = f"/pub/tanishs4/bubbleformer_logs/{model}_{job_id}/checkpoints/inference_rollouts/"
rollout_dir = pathlib.Path(rollout_dir)

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