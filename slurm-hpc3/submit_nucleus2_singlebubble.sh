#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu32
#SBATCH --job-name=train-nucleus-singlebubble
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=75GB
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

python scripts/train.py \
    model_cfg=nucleus2/nucleus2_divfree \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="float32" \
    pydataset=in_mem_forecast \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_staggered/ \
    data_cfg=singlebubble \
    normalizer_cfg=divfree \
    pydataset=in_mem_divfree_forecast \
    batch_size=16 \
    accumulate_grad_batches=1 \
    optim_cfg.params.lr=5e-4 \
    optim_cfg.params.weight_decay=1e-2 \
    max_steps=150000 \
    scheduler_cfg=trapezoidal \
    scheduler_cfg.params.warmup=4000 \
    scheduler_cfg.params.cooldown=40000 \
    log_dir=/pub/afeeney/nucleus_logs