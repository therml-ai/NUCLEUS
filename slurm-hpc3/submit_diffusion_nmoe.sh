#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --job-name=Nucleus
#SBATCH -o slurm-dnmoe-%x-%j.out
#SBATCH -e slurm-dnmoe-%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

cd /pub/tanishs4/NUCLEUS

# Fail loudly here rather than 20 minutes later inside torch. If the GPU cgroup
# wasn't set up for this job, nvidia-smi is already broken at this point.
echo "=== node: $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

echo "env setup"
uv venv "$TMPDIR/NUCLEUS"
source "$TMPDIR/NUCLEUS/bin/activate"
export PYTHONPYCACHE_DIR="$TMPDIR/pycache"
export PYTHONUNBUFFERED=1

echo "deps"
uv sync --no-cache --active --extra cu128
uv pip install -e .
uv pip install natten==0.21.5+torch2100cu128 -f https://whl.natten.org

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "diffusion start!"
python scripts/diffusion.py \
    checkpoint_path=/pub/tanishs4/bubbleformer_logs/neighbor_moe_poolboiling64_2026-09-25_56787438/checkpoints/last.ckpt \
    max_steps=30000 \
    log_dir=/pub/tanishs4/bubbleformer_logs/neighbor_moe_poolboiling64_2026-09-25_56787438/checkpoints