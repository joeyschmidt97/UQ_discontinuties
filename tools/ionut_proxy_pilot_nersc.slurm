#!/bin/bash
#SBATCH --job-name=uq-ionut
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-uq-ionut-%j.out

set -euo pipefail
: "${PSCRATCH:?PSCRATCH is not set}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="${UQ_ENV:-$PSCRATCH/conda-envs/uq-discontinuities}"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$REPO_ROOT"

for case in \
  ionut-itg-tem-3d-argmax-gamma ionut-itg-tem-3d-argmax-omega \
  ionut-itg-tem-3d-softmax-gamma ionut-itg-tem-3d-softmax-omega \
  ionut-itg-kbm-3d-argmax-gamma ionut-itg-kbm-3d-argmax-omega \
  ionut-itg-kbm-3d-softmax-gamma ionut-itg-kbm-3d-softmax-omega
do
  if [[ ! -d "data/3d/$case/seed-0" ]]; then
    python -m scripts.generate_ionut_slices --cases "$case" --output data
  fi
done
mkdir -p results/ionut-proxy-pilot

for dataset in \
  data/3d/ionut-itg-tem-3d-argmax-gamma/seed-0 \
  data/3d/ionut-itg-kbm-3d-argmax-gamma/seed-0 \
  data/6d/ionut-itg-tem-argmax-gamma/seed-0 \
  data/6d/ionut-itg-kbm-argmax-gamma/seed-0
do
  name=$(basename "$(dirname "$dataset")")
  for nu in 1.5 0.5; do
    tag=${nu/./}
    output="results/ionut-proxy-pilot/${name}-gpr-blend-m${tag}.json"
    if [[ ! -f "$output" ]]; then
      srun --ntasks=1 --cpus-per-task=2 python -m scripts.run_gp_experiment \
        --dataset "$dataset" --policy blend --uncertainty-weight 0.5 \
        --nu "$nu" --budget 256 --output "$output"
    fi
  done
done
