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

# BubbleML files converted with boiling-data's scripts/flashx_to_bubbleml.py, and the
# statistics.json that boiling-data's scripts/bubbleml_statistics.py wrote for them
DATA_DIR=  # TODO: the converted data
TRAIN_PATHS="[$DATA_DIR/TODO.hdf5]"
VAL_PATHS="[$DATA_DIR/TODO.hdf5]"

python scripts/train/train_nucleus2_moe.py \
    data_dir=$DATA_DIR \
    "data_cfg.train_paths=$TRAIN_PATHS" \
    "data_cfg.val_paths=$VAL_PATHS" \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="float32" \
    data.batch_size=4 \
    loop.accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=1e-3 \
    scheduler_cfg=trapezoidal \
    loop.max_steps=100000 \
    scheduler_cfg.params.warmup=5000 \
    scheduler_cfg.params.cooldown=10000 \
    log_dir=/pub/afeeney/nucleus2_logs
