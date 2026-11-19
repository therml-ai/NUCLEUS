#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu32
#SBATCH --job-name=train-nucleus-exp
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:RTX6000:1
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

# a directory made by boiling-data's scripts/split_dataset.py, holding the
# statistics.json that its scripts/bubbleml_statistics.py wrote for the train cases
SPLIT_DIR=  # TODO: the split converted data

python scripts/train/train_nucleus2_moe_divfree_projection.py \
    split_dir=$SPLIT_DIR \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="float32" \
    data.batch_size=64 \
    loop.accumulate_grad_batches=2 \
    optim_cfg.params.lr=1e-3 \
    optim_cfg.params.weight_decay=1e-3 \
    loop.max_steps=100000 \
    scheduler_cfg=trapezoidal \
    scheduler_cfg.params.warmup=2000 \
    scheduler_cfg.params.cooldown=20000 \
    log_dir=/pub/afeeney/nucleus_logs
