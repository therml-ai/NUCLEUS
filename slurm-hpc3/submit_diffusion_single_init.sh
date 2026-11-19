#!/bin/bash
#SBATCH -A AMOWLI_LAB
#SBATCH -p standard
#SBATCH --job-name=diff-single-init
#SBATCH -o /data/homezvol1/tanishs4/%x-%j.out
#SBATCH -e /data/homezvol1/tanishs4/%x-%j.err
#SBATCH --nodes=1
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

sbatch --job-name=diff-single-preflight \
    --cpus-per-task=2 --mem=6G --time=00:30:00 \
    --output=diff-single-preflight-%j.out \
    --wrap='bash /pub/tanishs4/NUCLEUS/slurm-hpc3/submit_diffusion_single.sh --preflight' \
    --error=diff-single-preflight-%j.err
