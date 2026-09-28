#!/bin/bash
#SBATCH -A amowli_lab_gpu
#SBATCH -p free-gpu32
#SBATCH --job-name=train-nucleus-flowboiling
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

python scripts/train.py \
    model_cfg=nucleus2/nucleus2_experiment \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="float32" \
    data_cfg=flowboiling \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/ \
    pydataset="forecast" \
    normalizer_cfg=standard \
    batch_size=4 \
    accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=1e-3 \
    scheduler_cfg=trapezoidal \
    scheduler_cfg.params.warmup=2000 \
    scheduler_cfg.params.cooldown=2000 \
    log_dir=/pub/afeeney/nucleus_logs