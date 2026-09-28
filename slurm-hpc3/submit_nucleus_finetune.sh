#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu
#SBATCH --job-name=finetune-nucleus
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
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
uv sync --no-cache --active --extra cu130
uv pip install -e .

CKPT_PATH=/pub/afeeney/nucleus_logs/nucleus2_moe_poolboiling64_2026-06-22_53641953/checkpoints/last.ckpt

python scripts/train.py \
    max_steps=10000 \
    model_cfg=nucleus2/nucleus2_experiment \
    checkpoint_path=$CKPT_PATH \
    data_cfg=poolboiling64 \
    normalizer_cfg=standard \
    batch_size=4 \
    optim_cfg.params.lr=1e-3 \
    optim_cfg.params.weight_decay=1e-2 \
    scheduler_cfg=cosine_warmup \
    scheduler_cfg.params.warmup=20 \
    scheduler_cfg.params.eta_min=1e-6 \
    log_dir=/pub/afeeney/nucleus_logs