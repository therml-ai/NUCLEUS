#!/bin/bash

#SBATCH --job-name=Nucleus       ## Name of the job.
#SBATCH -A AMOWLI_LAB      ## CHANGE account to charge
#SBATCH --partition=free-gpu32
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=slurm-%J.err  ## error log file
#SBATCH --output=slurm-%J.out ## output log file
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

cd /pub/tanishs4/NUCLEUS
git pull
uv venv $TMPDIR/NUCLEUS
source $TMPDIR/NUCLEUS/bin/activate
export PYTHONPYCACHE_DIR=pycache/
uv sync --no-cache --active --extra cu128
uv pip install -e .
uv pip install natten==0.21.5+torch2100cu128 -f https://whl.natten.org
python scripts/train.py \
    model_cfg=neighbor_moe/neighbor_moe_exp \
    data_cfg=poolboiling_single \
    normalizer_cfg=standard \
    model_cfg.params.patch_size=16 \
    batch_size=64 \
    val_check_interval=2653 \
    log_dir=/pub/tanishs4/bubbleformer_logs