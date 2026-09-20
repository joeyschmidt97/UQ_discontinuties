# Full local 2D benchmark provenance

- **Run date:** 2026-09-19
- **Source branch:** `codex/benchmark-2d`
- **Source commit:** `0cde444` (`Prepare NERSC runs and Ionut 3D slices`)
- **Platform:** local Windows Python environment
- **Result status:** complete; 33,396 of 33,396 expected trajectory rows succeeded

## Command

```powershell
.\.venv\Scripts\python.exe -m benchmark2d --cases smooth two-plane-four-peaks three-plane-three-peaks two-plane-asymmetric --arms grid sglib triangles gpr-var gpr-grad gpr-blend gpr-m05-var gpr-m05-grad gpr-m05-blend vwrs vurs --seeds 0 1 2 --budgets 64 128 192 256 --test-size 16384 --output results\2d-full-local-2026-09-19
```

The runner evaluates every integer paid-point count from 4 through 256. The four values passed to `--budgets` are report and checkpoint caps, not the only scored point counts.

## Scope

This report contains all eleven benchmark arms runnable in the local environment, across four surfaces and three geometry seeds. SG++ is absent because `pysgpp` is unavailable locally. Ionut's `sglib` arm ran successfully. No SG++ values were substituted or mocked, so this is an eleven-arm local result rather than the complete twelve-arm NERSC protocol.

The six rendered sheets, raw `results.json`, and `aggregate-scores.json` were produced together from the same completed run. The render manifest records the experiment source and renderer hashes.

## Validation

- Expected rows: `11 arms × 4 cases × 3 seeds × 253 point counts = 33,396`.
- Successful rows: 33,396; failed rows: 0.
- Cases: `smooth`, `two-plane-four-peaks`, `three-plane-three-peaks`, `two-plane-asymmetric`.
- Seeds: 0, 1, 2.
- Maximum paid evaluations per test: 256.
- Independent reference points per test: 16,384.

The committed report should be interpreted through all error panels. At the final budget, triangles has the lowest pooled normalized RMS, while the Matérn-3/2 gradient GP has the lowest worst-case holistic score and the strongest peak/VWFD scores. The result therefore does not support a single winner across every criterion.
