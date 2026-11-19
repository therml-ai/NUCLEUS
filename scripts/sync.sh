#!/usr/bin/env bash
# Copy this repo's source to hpc3 with rsync over ssh.
#
#   scripts/sync.sh [--dry-run]
#
# SYNC_HOST (default afeeney@access-hpc3.rcic.uci.edu) and SYNC_REMOTE_DIR (default
# /pub/$USER/project/NUCLEUS-fork) choose where the repo goes on the cluster.
set -euo pipefail

host="${SYNC_HOST:-afeeney@access-hpc3.rcic.uci.edu}"
remote_dir="${SYNC_REMOTE_DIR:-/pub/${USER}/project/NUCLEUS-fork}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .git, the environment and caches are rebuilt on the cluster; run outputs and
# local data files would be large, and the cluster has its own
excludes=(
    --exclude .git/
    --exclude .venv/
    --exclude __pycache__/
    --exclude .pytest_cache/
    --exclude .ruff_cache/
    --exclude .mypy_cache/
    --exclude .DS_Store
    --exclude /.claude/
    --exclude /outputs/
    --exclude /output/
    --exclude /wandb/
    --exclude '*.hdf5'
    --exclude '/*.json'
)

rsync_options=(--archive --compress --human-readable --itemize-changes)
for argument in "$@"; do
    case "${argument}" in
        --dry-run) rsync_options+=(--dry-run) ;;
        *) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
    esac
done

# rsync creates the repo directory itself, since the transfer node rejects plain
# ssh commands like mkdir; it creates only that last level, so its parent must exist
rsync "${rsync_options[@]}" "${excludes[@]}" "${repo_dir}/" "${host}:${remote_dir}/"
