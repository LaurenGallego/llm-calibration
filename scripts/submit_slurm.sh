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
sbatch scripts/slurm.sbatch "$@"
