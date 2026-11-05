#!/bin/bash
#SBATCH -A shrujans
#SBATCH -p free-gpu
#SBATCH --job-name=train-nucleus-continuity-loss
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=75GB
#SBATCH --gres=gpu:A30:1
#SBATCH --time=12:59:00

uv venv $TMPDIR/NUCLEUS
source $TMPDIR/NUCLEUS/bin/activate
# This make it so `__pycache__` files also go on the compute node.
export PYTHONPYCACHE_DIR=$TMPDIR/pycache/
# 1. `--no-cache` makes sure uv doesn't cache things in $HOME.
# 2. --active syncs to the currently activated environment. Otherwise, uv
#    tries to make another environment in the current directory.
# 3. --extra just gets stuff in pyproject.toml optional-dependencies
uv sync --no-cache --active --extra cu130
uv pip install -e .

# Set your wandb API key here before submitting
export WANDB_API_KEY=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

# a directory made by boiling-data's scripts/split_dataset.py, holding the
# statistics.json that its scripts/bubbleml_statistics.py wrote for the train cases
SPLIT_DIR=/share/crsp/lab/amowli/share/Bubbleml3/singlebubble-fc72-twall

python scripts/train/train_nucleus2_moe_continuity_loss.py \
    split_dir=$SPLIT_DIR \
    model_cfg.params.processor_blocks=12 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=2048 \
    model_cfg.params.patch_size=4 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="bfloat16" \
    data.batch_size=16 \
    data.in_memory=true \
    loop.accumulate_grad_batches=1 \
    loop.max_steps=20000 \
    log_dir=/pub/shrujans/nucleus_logs
