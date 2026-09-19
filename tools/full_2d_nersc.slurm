#!/bin/bash
#SBATCH --job-name=uq2d
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-uq2d-%j.out

set -euo pipefail
: "${PSCRATCH:?PSCRATCH is not set}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="${UQ_ENV:-$PSCRATCH/conda-envs/uq-discontinuities}"
export SG_LIB_PATH="${SG_LIB_PATH:-$PSCRATCH/src/sensitivity-driven-sparse-grid-approx}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$REPO_ROOT"

python -c "import pysgpp; from arms.sglib_arm import HAVE_SG_LIB; assert HAVE_SG_LIB"
srun --ntasks=1 --cpus-per-task=2 python -m benchmark2d \
  --seeds 0 1 2 \
  --budgets 64 128 192 256 \
  --test-size 16384 \
  --output results/2d-full-2026-09-19
