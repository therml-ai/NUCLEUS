#!/bin/bash
set -uo pipefail

checkpoint_path="${DIFFUSION_RESUME_PATH:-}"
log_dir=""
checkpoint_argument=false
for argument in "$@"; do
    case "$argument" in
        log_dir=*) log_dir="${argument#*=}" ;;
        +diffusion_resume_path=*|diffusion_resume_path=*)
            checkpoint_path="${argument#*=}"
            checkpoint_argument=true
            ;;
    esac
done
if [[ -z "$checkpoint_path" ]]; then
    if [[ -z "$log_dir" || -z "${SLURM_JOB_ID:-}" ]]; then
        echo "Set DIFFUSION_RESUME_PATH or supply log_dir inside a Slurm job." >&2
        exit 1
    fi
    checkpoint_path="$log_dir/diffusion_${SLURM_JOB_ID}/last.pt"
fi
if [[ "$checkpoint_argument" == false ]]; then
    set -- "$@" "+diffusion_resume_path=$checkpoint_path"
fi

child_pid=""
forward_preemption() {
    if [[ -n "$child_pid" ]]; then
        kill -USR1 "$child_pid" 2>/dev/null || true
    fi
}
trap forward_preemption TERM USR1

"$@" &
child_pid=$!
# A trapped signal interrupts wait even while the training process is alive.
while true; do
    wait "$child_pid"
    status=$?
    if ! kill -0 "$child_pid" 2>/dev/null; then
        break
    fi
done
if [[ "$status" -eq 75 && -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ ! -f "$checkpoint_path" ]]; then
        echo "Cannot requeue: missing diffusion checkpoint $checkpoint_path" >&2
        exit 1
    fi
    echo "Requeueing job $SLURM_JOB_ID with +diffusion_resume_path=$checkpoint_path"
    # Requeue reruns this batch script with the same job ID and checkpoint argument.
    scontrol requeue "$SLURM_JOB_ID" || exit 1
    exit 0
fi
exit "$status"
