#!/bin/bash

set -euo pipefail

launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$launcher_dir")"
export NUCLEUS_PROJECT_DIR="${NUCLEUS_PROJECT_DIR:-$repo_dir}"

exec bash "$launcher_dir/submit_diffusion_single.sh" --local-preflight "$@"
