#!/bin/bash

sbatch --job-name=diff-single-preflight \
    --cpus-per-task=2 --mem=6G --time=00:30:00 \
    --output=diff-single-preflight-%j.out \
    --wrap='bash /pub/tanishs4/NUCLEUS/slurm-hpc3/submit_diffusion_single.sh --preflight' \
    --error=diff-single-preflight-%j.err
