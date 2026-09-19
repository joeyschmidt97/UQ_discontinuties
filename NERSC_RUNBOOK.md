# NERSC runbook

This runbook prepares Perlmutter for the complete 2D rematch and optional
Ionut-proxy pilots. The benchmark is CPU-only. Replace `<account>` with the
NERSC project that should be charged.

NERSC recommends a custom conda environment for Python workloads. Perlmutter
jobs must declare a CPU or GPU constraint, and the account should be explicit:

- https://docs.nersc.gov/development/languages/python/using-python-perlmutter/
- https://docs.nersc.gov/jobs/
- https://docs.nersc.gov/jobs/policy/

## 1. Put the branch on NERSC

Push the local branch from the workstation first, then on Perlmutter:

```bash
cd "$PSCRATCH"
git clone --branch codex/benchmark-2d https://github.com/joeyschmidt97/UQ_discontinuties.git
cd UQ_discontinuties
git status --short
git log -1 --oneline
```

## 2. Create the environment and obtain both sparse-grid backends

From the repository root:

```bash
bash tools/nersc_setup.sh
```

For later sessions, restore the environment variables with:

```bash
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PSCRATCH/conda-envs/uq-discontinuities"
export SG_LIB_PATH="$PSCRATCH/src/sensitivity-driven-sparse-grid-approx"
```

The setup installs the pinned `pysgpp==3.3.1` wheel and checks that SG++ imports.
It also checks out Ionut's sensitivity-driven sparse-grid repository at the
locally validated commit. If the PyPI wheel is unavailable for the current
Perlmutter Python/architecture, stop rather than running an eleven-arm field;
build SG++ with Python bindings or use a compatible container.

## 3. Short NERSC preflight

Use an interactive CPU allocation so imports and both sparse-grid arms are
tested on a compute node:

```bash
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint cpu --account <account>
srun --ntasks 1 python -c "import pysgpp; from arms.sglib_arm import HAVE_SG_LIB; print('pysgpp OK; sglib=', HAVE_SG_LIB)"
srun --ntasks 1 python -m pytest tests/test_benchmark2d.py tests/test_exact_sparse.py tests/test_sgpp_real.py -q
exit
```

Do not launch the full run if either sparse-grid backend is skipped or the tests
fail.

## 4. Submit the frozen full 2D protocol

```bash
sbatch --account <account> tools/full_2d_nersc.slurm
```

The job runs all four surfaces, all twelve default arms, seeds 0/1/2, every
integer point count through 256, and 16,384 fixed evaluation points. Output is
written to `results/2d-full-2026-09-19`; the tracked N=150 verification remains
untouched. Monitor with `squeue -u "$USER"` and inspect `slurm-uq2d-<jobid>.out`.

After completion:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('results/2d-full-2026-09-19/results.json')
d = json.loads(p.read_text())
bad = [r for r in d['rows'] if r['status'] != 'ok']
print('rows:', len(d['rows']), 'bad:', len(bad), 'source:', d['source_hash'])
if bad:
    for row in bad[:20]: print(row['case'], row['seed'], row['arm'], row['status'], row.get('reason'))
    raise SystemExit(1)
PY
git status --short
git add results/2d-full-2026-09-19
git commit -m "Add full 2D twelve-arm benchmark"
git push origin codex/benchmark-2d
```

## 5. Ionut transition proxies

The repository already tracks nine **native 6D** archives: ITG–TEM and ITG–KBM
with argmax/softmax selection and gamma/omega outputs, plus a smooth
stellarator-style ITG/KBM proxy. They are phenomenological upstream formulas,
not gyrokinetic simulations.

The optional 3D generator creates conditional slices without refitting or
compressing those formulas:

- ITG–TEM varies `(RLTi, RLTe, nu)` and fixes `RLn=2`, `beta=0`, `ky=0.30`.
- ITG–KBM varies `(RLTi, beta, ky_scale)` and fixes `RLTe=5`, `RLn=2`, `nu=0`.

Generate all hard/soft gamma/omega slices:

```bash
python -m scripts.generate_ionut_slices --output data
```

Run the 50/50 GP uncertainty/gradient policy on the hard gamma transitions:

```bash
sbatch --account <account> tools/ionut_proxy_pilot_nersc.slurm
```

The submitted job executes the following reproducible loop on a CPU compute
node and skips already-completed JSON files if it is restarted:

```bash
mkdir -p results/ionut-proxy-pilot
for dataset in \
  data/3d/ionut-itg-tem-3d-argmax-gamma/seed-0 \
  data/3d/ionut-itg-kbm-3d-argmax-gamma/seed-0 \
  data/6d/ionut-itg-tem-argmax-gamma/seed-0 \
  data/6d/ionut-itg-kbm-argmax-gamma/seed-0
do
  name=$(basename "$(dirname "$dataset")")
  python -m scripts.run_gp_experiment \
    --dataset "$dataset" --policy blend --uncertainty-weight 0.5 \
    --nu 1.5 --budget 256 \
    --output "results/ionut-proxy-pilot/${name}-gpr-blend-m15.json"
  python -m scripts.run_gp_experiment \
    --dataset "$dataset" --policy blend --uncertainty-weight 0.5 \
    --nu 0.5 --budget 256 \
    --output "results/ionut-proxy-pilot/${name}-gpr-blend-m05.json"
done
```

This proxy command tests the GP blend only. The present VWRS/VURS acquisition
uses a 2D Delaunay gradient, and the common 2D grade uses triangulation. Neither
is silently relabeled as a valid 3D/6D method. A general-dimensional local-
variation estimator and common scorer must be validated before comparing
VWRS/VURS or declaring a proxy winner. Likewise, the current frozen-data runner
does not yet expose sg_lib or SG++; the full 2D job is the sparse-grid rematch.

Commit generated proxy data/results separately from the full 2D result so their
different scorers and scientific status remain explicit.
