#!/bin/bash
set -euo pipefail

: "${PSCRATCH:?PSCRATCH is not set; run this on NERSC}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX="${UQ_ENV:-$PSCRATCH/conda-envs/uq-discontinuities}"
SGLIB_ROOT="${SG_LIB_PATH:-$PSCRATCH/src/sensitivity-driven-sparse-grid-approx}"

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "$ENV_PREFIX" ]]; then
  mkdir -p "$(dirname "$ENV_PREFIX")"
  conda create --prefix "$ENV_PREFIX" python=3.11 pip -y
fi
conda activate "$ENV_PREFIX"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements-sgpp.txt"

if [[ ! -d "$SGLIB_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$SGLIB_ROOT")"
  git clone https://github.com/ionutfarcas/sensitivity-driven-sparse-grid-approx.git "$SGLIB_ROOT"
fi
git -C "$SGLIB_ROOT" fetch origin
git -C "$SGLIB_ROOT" checkout d13bc4661cbd3901a70190b52c477e7606576fa3
export SG_LIB_PATH="$SGLIB_ROOT"

cd "$REPO_ROOT"
python - <<'PY'
import pysgpp
from arms.sgpp_arm import HAVE_PYSGPP
from arms.sglib_arm import HAVE_SG_LIB
assert HAVE_PYSGPP and HAVE_SG_LIB
print("SG++ and sg_lib imports passed")
PY
python -m pytest tests/test_benchmark2d.py tests/test_datasets.py tests/test_ionut_data.py -q

printf '\nEnvironment ready. For later sessions:\n'
printf '  conda activate %q\n' "$ENV_PREFIX"
printf '  export SG_LIB_PATH=%q\n' "$SGLIB_ROOT"
