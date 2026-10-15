#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --job-name=Nucleus
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=24:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 CHECKPOINT_PATH [Hydra overrides...]" >&2
    exit 1
fi
diffusion_checkpoint_path="$1"
diffusion_log_dir="$(dirname "$(dirname "$diffusion_checkpoint_path")")/diffusion"
shift

cd /pub/tanishs4/NUCLEUS

echo "env setup"
uv venv "$TMPDIR/NUCLEUS"
source "$TMPDIR/NUCLEUS/bin/activate"
export PYTHONPYCACHE_DIR="$TMPDIR/pycache"
export PYTHONUNBUFFERED=1

echo "deps"
uv sync --no-cache --active --extra cu130

uv pip check
python -c 'import torchvision; from diffusers import UNet2DModel; import natten; print("torchvision", torchvision.__version__, "natten", natten.__version__)'

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "diffusion start!"
python scripts/diffusion.py \
    model_cfg=nucleus2/nucleus2_experiment \
    model_cfg.params.patch_size=16 \
    data_cfg=poolboiling \
    normalizer_cfg=standard \
    pydataset=in_mem_forecast \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/ \
    checkpoint_path="$diffusion_checkpoint_path" \
    history_time_window=8 \
    future_time_window=8 \
    time_step=1 \
    batch_size=4 \
    max_steps=6 \
    log_dir="$diffusion_log_dir" \
    "$@"
