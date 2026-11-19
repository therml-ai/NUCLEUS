#!/bin/bash
#SBATCH -A AMOWLI_LAB
#SBATCH -p free-gpu32
#SBATCH --job-name=diff-single
#SBATCH -o /data/homezvol1/tanishs4/%x-%j.out
#SBATCH -e /data/homezvol1/tanishs4/%x-%j.err
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --exclude=hpc3-gpu-n54-00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=12:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail
trap 'exit_code=$?; echo "Job ${SLURM_JOB_ID:-unknown} failed at line $LINENO (exit $exit_code)" >&2; exit "$exit_code"' ERR
export HYDRA_FULL_ERROR=1
echo "Starting job ${SLURM_JOB_ID:-unknown} on $(hostname)"

preflight=false
if [[ "${1:-}" == --preflight ]]; then
    preflight=true
    shift
    export CUDA_VISIBLE_DEVICES=""
    echo "Running single-bubble diffusion preflight on CPU"
fi
if (( $# )); then
    echo "Usage: bash $0 [--preflight]" >&2
    exit 2
fi

cd /pub/tanishs4/NUCLEUS

echo "env setup"
scratch_dir="${TMPDIR:-/tmp}"
environment_dir="$scratch_dir/NUCLEUS"
if "$preflight"; then
    environment_dir="$scratch_dir/NUCLEUS-preflight-${USER}-${SLURM_JOB_ID:-manual}"
fi
uv venv "$environment_dir"
source "$environment_dir/bin/activate"
export PYTHONPYCACHEPREFIX="$environment_dir/pycache"
export PYTHONUNBUFFERED=1

echo "deps"
uv sync --no-cache --active --extra cu128

uv pip check
python -c 'import torchvision; from diffusers import UNet2DModel; import natten; print("torchvision", torchvision.__version__, "natten", natten.__version__)'
diffusion_args=(
    model_cfg=nucleus2/nucleus2_experiment
    model_cfg.params.processor_blocks=8
    model_cfg.params.embed_dim=512
    model_cfg.params.num_experts=6
    model_cfg.params.moe_intermediate_dim=1024
    model_cfg.params.patch_size=16
    model_cfg.params.patching=Linear
    model_cfg.params.activation_dtype=float32
    data_cfg=singlebubble
    normalizer_cfg=standard
    pydataset=in_mem_forecast
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/
    checkpoint_path=/pub/tanishs4/bubbleformer_logs/nucleus2_moe_singlebubble_2026-10-08_57936970/checkpoints/last.ckpt
    history_time_window=8
    future_time_window=8
    time_step=1
    batch_size=4
    max_steps=10000
    log_dir=/pub/tanishs4/bubbleformer_logs/nucleus2_moe_singlebubble_2026-10-08_57936970
)

if "$preflight"; then
    exec python scripts/diffusion.py "${diffusion_args[@]}" +diffusion_preflight=true
fi

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
echo "diffusion start!"
exec bash slurm-hpc3/run_diffusion_requeue.sh python scripts/diffusion.py "${diffusion_args[@]}"
