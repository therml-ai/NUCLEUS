#!/bin/bash
#SBATCH -A AMOWLI_LAB 
#SBATCH -p free-gpu32
#SBATCH --job-name=Nucleus
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

cd /pub/tanishs4/NUCLEUS

echo "env setup"
uv venv "$TMPDIR/NUCLEUS"
source "$TMPDIR/NUCLEUS/bin/activate"
export PYTHONPYCACHE_DIR="$TMPDIR/pycache"
export PYTHONUNBUFFERED=1

echo "deps"
uv sync --no-cache --active --extra cu128

uv pip check
python -c 'import torchvision; from diffusers import UNet2DModel; import natten; print("torchvision", torchvision.__version__, "natten", natten.__version__)'

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "diffusion start!"
python scripts/diffusion.py \
    checkpoint_path=/pub/tanishs4/bubbleformer_logs/neighbor_moe_poolboiling64_2026-09-25_56787438/checkpoints/last.ckpt \
    max_steps=30000 \
    log_dir=/pub/tanishs4/bubbleformer_logs/neighbor_moe_poolboiling64_2026-09-25_56787438/checkpoints

#may have to un-hardcode this oops