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

python scripts/inf.py \
    model_cfg=nucleus2/nucleus2_experiment \
    checkpoint_path=/share/crsp/lab/amowli/share/nucleus2-model-ckpts/neighbor_moe_exp_may13.ckpt \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/ \
    data_cfg=poolboiling64 \
    normalizer_cfg=standard \
    log_dir=/pub/afeeney/nucleus_logs/ \
    trajectory_steps=1500 \
