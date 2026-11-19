#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --job-name=train-nucleus-singlebubble
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=75GB
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
SPLIT_DIR=/share/crsp/lab/amowli/share/Bubbleml3/singlebubble-fc72-twall

python scripts/train/train_nucleus2_moe_divfree_projection.py \
    split_dir=$SPLIT_DIR \
    model_cfg.params.processor_blocks=6 \
    model_cfg.params.embed_dim=256 \
    model_cfg.params.num_experts=4 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="bfloat16" \
    data.batch_size=16 \
    data.in_memory=true \
    loop.accumulate_grad_batches=1 \
    optim_cfg.params.lr=5e-4 \
    optim_cfg.params.weight_decay=1e-2 \
    loop.max_steps=150000 \
    scheduler_cfg=trapezoidal \
    scheduler_cfg.params.warmup=4000 \
    scheduler_cfg.params.cooldown=40000 \
    log_dir=/pub/afeeney/nucleus_logs
