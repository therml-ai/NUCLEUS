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

# BubbleML files converted with boiling-data's scripts/flashx_to_bubbleml.py, and the
# statistics.json that boiling-data's scripts/bubbleml_statistics.py wrote for them
DATA_DIR=  # TODO: the converted data
TEST_PATHS="[$DATA_DIR/TODO.hdf5]"

# rollouts are written next to the checkpoint, in a rollouts directory
CKPT=  # TODO: a last.pt or best.pt written by scripts/train/train_nucleus2_moe.py

python scripts/rollout.py \
    checkpoint_path=$CKPT \
    data_dir=$DATA_DIR \
    "data_cfg.test_paths=$TEST_PATHS" \
    trajectory_steps=1500
