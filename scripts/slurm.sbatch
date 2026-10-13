#!/bin/bash
#SBATCH --job-name=madcal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "host $(hostname) | commit $(git rev-parse --short HEAD) | configs: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/madcal run "$@"
