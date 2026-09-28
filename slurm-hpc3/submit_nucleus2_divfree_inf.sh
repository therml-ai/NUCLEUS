#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu
#SBATCH --job-name=nucleus2-inference
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A30:1
#SBATCH --time=00:10:00

uv venv $TMPDIR/NUCLEUS
source $TMPDIR/NUCLEUS/bin/activate
# This make it so `__pycache__`` files also go on the compute node.
export PYTHONPYCACHE_DIR=$TMPDIR/pycache/
# 1. `--no-cache` makes sure uv doesn't cache things in $HOME.
# 2. --active syncs to the currently activated environment. Otherwise, uv
#    tries to make another environment in the current directory.
# 3. --extra just gets stuff in pyproject.toml optional-dependencies
uv sync --no-cache --active --extra cu130
uv pip install -e .

uv pip install -e ../boiling-viz/

CKPT=/pub/afeeney/nucleus_logs/nucleus2_moe_divfree_singlebubble_2026-09-03_55730388/checkpoints/last.ckpt
ROLLOUT=/pub/afeeney/nucleus_logs/nucleus2_moe_divfree_singlebubble_2026-09-03_55730388/checkpoints/rollouts/saturated_fc72_100/

python scripts/inf.py \
    model_cfg=nucleus2/nucleus2_divfree \
    checkpoint_path=$CKPT \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_staggered/ \
    data_cfg=singlebubble \
    normalizer_cfg=divfree \
    log_dir=/pub/afeeney/nucleus_logs/ \
    trajectory_steps=300 \

python scripts/visualize_rollout.py --path $ROLLOUT \