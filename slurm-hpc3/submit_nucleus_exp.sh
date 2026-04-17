#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu
#SBATCH --job-name=train-nucleus-exp
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:A30:1
#SBATCH --time=12:59:00

uv venv $TMPDIR/NUCLEUS
source $TMPDIR/NUCLEUS/bin/activate
# This make it so `__pycache__`` files also go on the compute node.
export PYTHONPYCACHE_DIR=$TMPDIR/pycache/
# 1. `--no-cache` makes sure uv doesn't cache things in $HOME.
# 2. --active syncs to the currently activated environment. Otherwise, uv
#    tries to make another environment in the current directory.
# 3. --extra just gets stuff in pyproject.toml optional-dependencies
uv sync --no-cache --active --extra cu128
uv pip install -e .
uv pip install natten==0.21.5+torch2100cu128 -f https://whl.natten.org

python scripts/train.py \
    model_cfg=neighbor_moe/neighbor_moe_exp \
    data_cfg=poolboiling \
    normalizer_cfg=standard \
    model_cfg.params.patch_size=16 \