#!/bin/bash
#SBATCH -A AMOWLI_LAB
#SBATCH -p free-gpu32
#SBATCH --job-name=train-nucleus-exp
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --exclude=hpc3-gpu-n54-00
#SBATCH --cpus-per-task=12
#SBATCH --mem=24GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=48:00:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

set -euo pipefail

project_dir="${NUCLEUS_PROJECT_DIR:-/pub/tanishs4/NUCLEUS}"
if [[ ! -f "$project_dir/pyproject.toml" || ! -f "$project_dir/scripts/train.py" ]]; then
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
python -c 'import torchvision; import natten; print("torchvision", torchvision.__version__, "natten", natten.__version__)'
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

echo "training start!"
python scripts/train.py \
    model_cfg=nucleus2/nucleus2_experiment \
    model_cfg.params.processor_blocks=8 \
    model_cfg.params.embed_dim=512 \
    model_cfg.params.num_experts=6 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.patch_size=16 \
    model_cfg.params.patching="Linear" \
    model_cfg.params.activation_dtype="float32" \
    data_cfg=poolboiling \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/ \
    pydataset="forecast" \
    normalizer_cfg=standard \
    batch_size=4 \
    accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=1e-3 \
    scheduler_cfg=trapezoidal \
    max_steps=100000 \
    scheduler_cfg.params.warmup=5000 \
    scheduler_cfg.params.cooldown=10000 \
    log_dir=/pub/tanishs4/nucleus2_logs \
    profile=False \
    "$@"
