# results/2d — provenance

Two-dimensional benchmark: 4 surfaces x 3 seeds, every integer N from 4 to 256,
Delaunay reconstruction scored against a fixed 16,384-point Sobol set.

Code: `benchmark2d/` (+ `arms/` for the external sparse-grid backends).

## Commands

```bash
# original seven arms, one trajectory per case/seed
python -m benchmark2d --seeds 0 1 2 --budgets 32 64 128 256 --native-test-size 1024 --output outputs/full

# the five GP uncertainty/gradient ratio arms, one output directory each
for arm in gpr-u20-g80 gpr-u30-g70 gpr-u50-g50 gpr-u70-g30 gpr-u80-g20; do
  python -m benchmark2d --arms $arm --seeds 0 1 2 --budgets 32 64 128 256     --native-test-size 1024 --output outputs/ratio-$arm
done

# join them and keep the best three blends, re-rendering this directory in place
cp results/2d/results.json /tmp/base.json
python -m benchmark2d.merge --base /tmp/base.json --add outputs/ratio-*/results.json   --top 3 --output results/2d
```

## Contents

| File | Produced by |
|---|---|
| `results.json` | the runs above, joined by `benchmark2d.merge` |
| `aggregate-scores.json`, `figures/`, `index.html`, `README.md` | `benchmark2d.report.render`, called by both commands |
| `render-manifest.json` | renderer hash and figure list |
| `*-experiment-sources.zip` | source snapshots of the code that produced each batch |

## Source hashes

- `29a57bb2d7143bcf66c84942f88a646a4b368382a9e3b86ae31cc94e801324d3`
- `cb1eb0a7d95655e7f57736d218dc47b459b7c8cd85a97a86723e95390ca4aa1f`

Executed on Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.43, Python 3.11.16, {'numpy': '1.26.4', 'scipy': '1.17.1', 'scikit-learn': '1.9.0', 'matplotlib': '3.11.1'}.
Arms in this file: grid, moe, sglib, sgpp, gpr-var, gpr-grad, triangles, gpr-u50-g50, gpr-u70-g30, gpr-u30-g70.
