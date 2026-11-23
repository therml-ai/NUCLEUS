#!/bin/bash
#SBATCH -A AMOWLI_LAB
#SBATCH -p standard
#SBATCH --job-name=diff-single-init
#SBATCH -o /data/homezvol1/tanishs4/%x-%j.out
#SBATCH -e /data/homezvol1/tanishs4/%x-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --mail-user=tanishs4@uci.edu
#SBATCH --mail-type=BEGIN,END

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch "$0" "$@"
fi

exec bash /pub/tanishs4/NUCLEUS/slurm-hpc3/submit_diffusion_single.sh --preflight "$@"
