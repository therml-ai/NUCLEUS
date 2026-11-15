#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --job-name=diff-nmoe
#SBATCH -o /data/homezvol1/tanishs4/%x-%j.out
#SBATCH -e /data/homezvol1/tanishs4/%x-%j.err
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=24:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail
trap 'exit_code=$?; echo "Job ${SLURM_JOB_ID:-unknown} failed at line $LINENO (exit $exit_code)" >&2; exit "$exit_code"' ERR
export HYDRA_FULL_ERROR=1
echo "Starting job ${SLURM_JOB_ID:-unknown} on $(hostname)"

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
uv sync --no-cache --active --extra cu130

uv pip check
python -c 'import torchvision; from diffusers import UNet2DModel; import natten; print("torchvision", torchvision.__version__, "natten", natten.__version__)'

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "diffusion start!"
exec bash slurm-hpc3/run_diffusion_requeue.sh python scripts/diffusion.py \
    model_cfg=nucleus2/nucleus2_experiment \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching=Linear \
    model_cfg.params.activation_dtype=float32 \
    data_cfg=poolboiling \
    normalizer_cfg=standard \
    pydataset=in_mem_forecast \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/ \
    checkpoint_path=/pub/tanishs4/nucleus2_logs/nucleus2_moe_poolboiling512_2026-10-29_58609204/checkpoints/last.ckpt \
    history_time_window=8 \
    future_time_window=8 \
    time_step=1 \
    batch_size=4 \
    max_steps=100 \
    log_dir=/pub/tanishs4/nucleus2_logs/nucleus2_moe_poolboiling512_2026-10-29_58609204 \
    "$@"
