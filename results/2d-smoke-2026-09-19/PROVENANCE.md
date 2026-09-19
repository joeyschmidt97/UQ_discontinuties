# 2-D N=150 holistic-metrics verification

This is an expanded implementation check, not the final benchmark ranking. It
uses one geometry seed and the two revised folded reference surfaces, scoring
every integer point count from 4 through 150 on 4,096 fixed evaluation points.

Run from the repository root with the repository `.venv`:

```powershell
.\.venv\Scripts\python.exe -m benchmark2d `
  --cases two-plane-four-peaks three-plane-three-peaks `
  --arms grid sglib triangles gpr-var gpr-grad gpr-blend gpr-m05-var gpr-m05-grad gpr-m05-blend vwrs vurs `
  --seeds 0 --budgets 50 100 150 --test-size 4096 `
  --output results\2d-smoke-2026-09-19
```

SG++ is omitted because this Windows environment does not have the `pysgpp`
Python bindings. An earlier all-default-arm preflight reported that arm as
unavailable. No SG++ values are mocked or inferred. Ionut's `sglib` completed.

All 60 focused tests passed after the report change:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_benchmark2d.py -q -p no:cacheprovider
```

Sheet 05 shows equal-weight pooled normalized RMS across the two surfaces.
Sheet 06 gives one pooled subplot per error family with every method overlaid;
it does not split the same error into separate manifold panels. At N=150,
the 50/50 Matérn-3/2 GP gradient/uncertainty blend has the lowest combined RMS
(0.0250). This one-seed, two-case result is not sufficient to select a winner.
