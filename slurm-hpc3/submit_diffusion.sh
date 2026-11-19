#!/bin/bash
#SBATCH -A AMOWLI_LAB
#SBATCH -p free-gpu32
#SBATCH --job-name=diffusion
#SBATCH -o /data/homezvol1/tanishs4/slurm-%x-%j.out
#SBATCH -e /data/homezvol1/tanishs4/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --cpus-per-task=6
#SBATCH --mem=36GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

project_dir="${NUCLEUS_PROJECT_DIR:-/pub/tanishs4/NUCLEUS}"
if [[ ! -f "$project_dir/pyproject.toml" || ! -f "$project_dir/scripts/diffusion.py" ]]; then
    echo "NUCLEUS project not found at $project_dir. Set NUCLEUS_PROJECT_DIR to the repository root." >&2
    exit 1
fi
cd "$project_dir"
echo "Running NUCLEUS from $PWD"

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
exec bash slurm-hpc3/run_diffusion_requeue.sh python scripts/diffusion.py \
    checkpoint_path=/pub/tanishs4/bubbleformer_logs/neighbor_moe_poolboiling64_2026-09-25_56787438/checkpoints/last.ckpt \
    max_steps=30000 \
    log_dir=/pub/tanishs4/bubbleformer_logs/neighbor_moe_poolboiling64_2026-09-25_56787438/checkpoints

#may have to un-hardcode this oops