# 2-D holistic-metrics smoke verification

This is a small implementation check, not a benchmark ranking. It uses one
geometry seed, two folded reference surfaces, and a maximum of 24 paid samples.
The report is tracked so the new metrics and all-model plots can be reviewed
before spending the full rematch budget.

Run from the repository root with the repository `.venv`:

```powershell
.\.venv\Scripts\python.exe -m benchmark2d `
  --cases two-plane-four-peaks three-plane-three-peaks `
  --arms grid sglib triangles gpr-var gpr-grad gpr-blend gpr-m05-var gpr-m05-grad gpr-m05-blend vwrs vurs `
  --seeds 0 --budgets 16 24 --test-size 2048 `
  --output results\2d-smoke-2026-09-19
```

An initial run requested all twelve default arms. SG++ was unavailable because
this Windows environment does not have the `pysgpp` Python bindings. No SG++
values are mocked or inferred. The report was regenerated with the eleven arms
that completed so its aggregate curve contains a fair matched field.

All 59 focused tests passed after generation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_benchmark2d.py -q -p no:cacheprovider
```

At N=24, triangles has the lowest pooled normalized RMS in this smoke field
(0.1125). The result is too small for winner selection: it has one seed, only
two of the four cases, and a much smaller cap than the planned rematch.
