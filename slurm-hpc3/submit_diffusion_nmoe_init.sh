#!/bin/bash

sbatch --job-name=diff-nmoe-preflight \
    --cpus-per-task=2 --mem=6G --time=00:30:00 \
    --output=diff-nmoe-preflight-%j.out \
    --error=diff-nmoe-preflight-%j.err \
    --wrap='bash "${NUCLEUS_PROJECT_DIR:-/pub/tanishs4/NUCLEUS}/slurm-hpc3/submit_diffusion_nmoe.sh" --preflight'
