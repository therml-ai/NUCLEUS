#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu
#SBATCH --job-name=nucleus2-inference
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A30:1
#SBATCH --time=00:59:00

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

# a directory made by boiling-data's scripts/split_dataset.py, holding the
# statistics.json that its scripts/bubbleml_statistics.py wrote for the train cases
SPLIT_DIR=  # TODO: the split converted data

# rollouts are written next to the checkpoint, in a rollouts directory
CKPT=  # TODO: a last.pt or best.pt written by scripts/train/train_nucleus2_moe.py

python scripts/rollout.py \
    checkpoint_path=$CKPT \
    model_cfg=nucleus2/nucleus2_experiment \
    split_dir=$SPLIT_DIR \
    trajectory_steps=1500
