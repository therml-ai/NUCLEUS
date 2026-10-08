#!/bin/bash
#SBATCH -A AMOWLI_LAB 
#SBATCH -p free-gpu32
#SBATCH --job-name=Nucleus
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

# if [[ $# -lt 3 ]]; then
#     echo "Usage: sbatch $0 CHECKPOINT_PATH DATA_DIR LOG_DIR [Hydra overrides...]" >&2
#     exit 1
# fi
# diffusion_checkpoint_path="$1"
# diffusion_data_dir="$2"
# diffusion_log_dir="$3"
# shift 3

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
    model_cfg=nucleus2/nucleus2_divfree \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching=Linear \
    model_cfg.params.activation_dtype=float32 \
    data_cfg=singlebubble \
    normalizer_cfg=divfree \
    pydataset=in_mem_forecast \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_staggered/ \
    checkpoint_path=/pub/tanishs4/bubbleformer_logs/nucleus2_moe_divfree_singlebubble_2026-10-01_57616757/checkpoints/last.ckpt \
    history_time_window=8 \
    future_time_window=8 \
    time_step=1 \
    batch_size=4 \
    max_steps=30000 \
    log_dir=/pub/tanishs4/bubbleformer_logs/diffusion_single
