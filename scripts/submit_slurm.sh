#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
    echo "missing .env: copy .env.example to .env and fill it in" >&2
    exit 1
fi
set -a
source .env
set +a
options=()
while [ $# -gt 0 ] && [[ "$1" == -* ]]; do
    options+=("$1")
    shift
done
sbatch ${options[@]+"${options[@]}"} scripts/slurm.sbatch "$@"
