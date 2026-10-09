"""Two registered checks on the 2026-10-07 surprise test (vault note
2026-10-07-vurs-surprise-scoring, "Follow-up checks -- protocol").

A. Scorer robustness. Label surprise recomputed with a GP classifier
   (`benchmarknd.surprise.gp_label_posterior`) in place of the Dirichlet-kNN
   vote, whose accuracy is tied to nearest-run distance and so may favour
   space-filling designs. Primary endpoint A1: GP-classifier label surprise,
   campaign mean (N = 23, 38, 61, 98). Holm across the 9 comparators per pool.
B. Region grading. Scores restricted to the pool regions the survey cares
   about, defined in `benchmarknd.pool.exploration_regions` before this test:
   the design transition band and the per-mode top-10% growth-rate peaks.
   Primary endpoints, campaign means: B1 band label surprise (kNN, bits),
   B2 band error (`region_design_band_nrmse`), B3 peak error
   (`region_peak_nrmse`), B4 peak growth-rate surprise (nats). Holm across
   9 comparators x 4 endpoints per pool.

Paired vurs - comparator differences over 20 seeds, 95% bootstrap CI,
Wilcoxon; margins 0.05 bits, 0.10 nats, and 5% of vurs's mean for errors.
Verdicts as in `scripts.surprise_board`.

    python -m scripts.surprise_checks score --runs results/confirm-2026-10-05 results/board20-2026-10-07 \
        --output results/surprise-checks-2026-10-09
    python -m scripts.surprise_checks analyse --root results/surprise-checks-2026-10-09
"""
import argparse
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmarknd.pool import exploration_regions, load_pool
from benchmarknd.surprise import surprise_scores
from scripts.surprise_board import COMPARATORS, POOLS, holm, paired, trajectories, verdict

CHECKPOINTS = (23, 38, 61, 98, 256)
CAMPAIGN = (23, 38, 61, 98)
REGIONS = ("design_band", "peak", "quiet")
PRIMARY = {
    "A": {"A1 label surprise, GP classifier": ("gpc_s_label_bits", .05, "abs")},
    "B": {"B1 band label surprise": ("design_band_s_label_bits", .05, "abs"),
          "B2 band error": ("region_design_band_nrmse", .05, "rel"),
          "B3 peak error": ("region_peak_nrmse", .05, "rel"),
          "B4 peak growth-rate surprise": ("peak_s_gamma_nats", .10, "abs")},
}
SECONDARY = {
    "A2 growth-rate surprise, GP-classifier mixture": ("gpc_s_gamma_nats", CAMPAIGN),
    "A1 at N=256": ("gpc_s_label_bits", (256,)),
    "band growth-rate surprise": ("design_band_s_gamma_nats", CAMPAIGN),
    "peak label surprise": ("peak_s_label_bits", CAMPAIGN),
    "quiet label surprise": ("quiet_s_label_bits", CAMPAIGN),
    "quiet growth-rate surprise": ("quiet_s_gamma_nats", CAMPAIGN),
    "quiet error": ("region_quiet_nrmse", CAMPAIGN),
    "B1 at N=256": ("design_band_s_label_bits", (256,)),
    "B2 at N=256": ("region_design_band_nrmse", (256,)),
    "B3 at N=256": ("region_peak_nrmse", (256,)),
    "B4 at N=256": ("peak_s_gamma_nats", (256,)),
}


def score_one(job):
    (arm, seed), (pool_path, sha, order, rows) = job
    oracle, meta = load_pool(pool_path)
    if meta["sha256"] != sha:
        raise ValueError("pool file changed since the run; refusing to score")
    regions = exploration_regions(oracle)
    out = []
    for n in CHECKPOINTS:
        paid = np.asarray(order[:n])
        row = dict(arm=arm, seed=seed, n=n)
        gpc = surprise_scores(oracle, paid, classifier="gpc")
        row.update(gpc_s_label_bits=gpc["s_label_bits"], gpc_s_gamma_nats=gpc["s_gamma_nats"],
                   gpc_label_entropy_bits=gpc["label_entropy_bits"])
        knn = surprise_scores(oracle, paid, return_points=True)
        for name in REGIONS:
            mask = regions[name][knn["held_index"]]
            row[f"{name}_s_label_bits"] = float(knn["label_bits"][mask].mean()) if mask.any() else None
            row[f"{name}_s_gamma_nats"] = float(knn["gamma_nats"][mask].mean()) if mask.any() else None
            row[f"region_{name}_nrmse"] = rows.get(n, {}).get(f"region_{name}_nrmse")
        out.append(row)
    return out


def score(args):
    for pool in POOLS:
        target = args.output/pool
        target.mkdir(parents=True, exist_ok=True)
        done = {p.stem for p in target.glob("*.json")}
        jobs = [job for job in sorted(trajectories(args.runs, pool).items())
                if f"{job[0][0]}_seed{job[0][1]}" not in done]
        print(f"{pool}: {len(jobs)} trajectories to score", flush=True)
        with ProcessPoolExecutor(args.workers) as executor:
            for rows in executor.map(score_one, jobs):
                arm, seed = rows[0]["arm"], rows[0]["seed"]
                (target/f"{arm}_seed{seed}.json").write_text(json.dumps(rows), encoding="utf-8")
                print(f"{pool} {arm} seed {seed}: A1 {rows[1]['gpc_s_label_bits']:.3f} at N=38", flush=True)


def analyse(args):
    rng = np.random.default_rng(20261009)
    report = dict(protocol=__doc__, pools={})
    for pool in POOLS:
        rows = []
        for path in sorted((args.root/pool).glob("*.json")):
            rows += json.loads(path.read_text())
        result = {}
        for test, endpoints in PRIMARY.items():
            tests, keys = {}, []
            for name, (key, margin, kind) in endpoints.items():
                for other in COMPARATORS:
                    t = paired(rows, other, key, CAMPAIGN, rng)
                    if t is None:
                        continue
                    t["margin"] = margin*abs(t["reference_mean"]) if kind == "rel" else margin
                    tests.setdefault(name, {})[other] = t
                    keys.append((name, other))
            for (name, other), p in zip(keys, holm(np.array([tests[n][o]["p"] for n, o in keys]))):
                tests[name][other]["p_holm"] = float(p)
                tests[name][other]["verdict"] = verdict(tests[name][other], tests[name][other]["margin"])
            result[test] = tests
        result["secondary"] = {name: {o: paired(rows, o, key, budgets, rng) for o in COMPARATORS}
                               for name, (key, budgets) in SECONDARY.items()}
        report["pools"][pool] = result

    def v(pool, test, name, other):
        return report["pools"][pool][test][name][other]["verdict"]
    claims = {}
    for other in COMPARATORS:
        a = [v(p, "A", "A1 label surprise, GP classifier", other) for p in POOLS]
        band = [any(v(p, "B", n, other) == "faster" for n in ("B1 band label surprise", "B2 band error"))
                for p in POOLS]
        peak = [any(v(p, "B", n, other) == "faster" for n in ("B3 peak error", "B4 peak growth-rate surprise"))
                for p in POOLS]
        worse = any(v(p, "B", n, other) == "slower" for p in POOLS for n in PRIMARY["B"])
        claims[other] = dict(A_verdicts=a, A_faster_both=all(x == "faster" for x in a),
                             B_concentrated=all(band) and all(peak) and not worse,
                             B_band=band, B_peak=peak, B_any_worse=worse)
    report["claims"] = claims
    out = args.root/"checks_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for pool in POOLS:
        print(f"\n{pool} (vurs - comparator, campaign mean; negative = vurs better)")
        for test in PRIMARY:
            for name, comps in report["pools"][pool][test].items():
                for other, t in comps.items():
                    print(f"  {name:32s} {other:14s} {t['mean']:+.4f} [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] "
                          f"ref {t['reference_mean']:.3f} p_holm={t['p_holm']:.3g} {t['verdict']}")
    print("\nclaims:")
    for other, c in claims.items():
        print(f"  {other:14s} A {c['A_verdicts']} faster-both={c['A_faster_both']} | "
              f"B band {c['B_band']} peak {c['B_peak']} worse={c['B_any_worse']} concentrated={c['B_concentrated']}")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--runs", type=pathlib.Path, nargs="+", required=True)
    s.add_argument("--output", type=pathlib.Path, required=True)
    s.add_argument("--workers", type=int, default=14)
    a = sub.add_parser("analyse")
    a.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    score(args) if args.command == "score" else analyse(args)


if __name__ == "__main__":
    main()
