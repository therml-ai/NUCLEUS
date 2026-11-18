#!/bin/bash
#SBATCH -A shrujans
#SBATCH -p free-gpu
#SBATCH --job-name=nucleus-sweep-agent
#SBATCH -o slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=75GB
#SBATCH --gres=gpu:A30:1
#SBATCH --time=12:59:00
#SBATCH --array=0-3   # one job per sweep run (4 weight values)

uv venv $TMPDIR/NUCLEUS
source $TMPDIR/NUCLEUS/bin/activate
export PYTHONPYCACHE_DIR=$TMPDIR/pycache/
uv sync --no-cache --active --extra cu130
uv pip install -e .

export WANDB_API_KEY=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

# TODO: replace with the sweep ID printed by:
#   wandb sweep config/sweep_continuity_loss_weight.yaml
SWEEP_ID=  # e.g. hpcforge/nucleus/abc12345

wandb agent --count 1 $SWEEP_ID
