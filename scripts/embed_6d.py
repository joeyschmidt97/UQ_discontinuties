"""Embed the native 6D Ionut proxy and read sampler placement on it.

    python -m scripts.embed_6d --output results/embedding-6d-<date>

Ground rules, from the VWRS/VURS note's sampling-space display section:

* Every distance, neighbourhood, segment and score is computed in the scaled
  feature space, never in the embedding. UMAP and PaCMAP are display only.
* One fixed probe cloud, embedded once, so methods and budgets share a frame.
* Two projections side by side, each with its neighbourhood preservation
  (trustworthiness) reported, before any island is read as structure.

What is embedded. A uniform 6D input box embeds as a featureless blob, so the
inputs alone cannot produce islands. Islands can appear only when the
embedding also sees the response. Features are the scaled inputs plus a
standardized response block multiplied by a response weight w:

    w = 0    pure parameter space
    w > 0    the response may pull points of one branch together

The displayed response block is (gamma, omega), what a GENE run reports. A
second block adding both branch growth rates is swept but never displayed:
the branch label is their argmax, so that view is circular by construction.

Whether islands form depends on the pair. In ITG-TEM the two branches rotate
in opposite directions, so omega's sign alone separates them; in ITG-KBM both
rotate negative and that cue is absent, which is the harder and more
realistic case.

The branch label (argmax of the branch growth rates) is ground truth here,
which is the point of running this on a proxy before real GENE data: every
tool can be checked against a known answer.

Tools ported from the MGKDB fingerprint study (fmc.analysis.concordance):

* kNN branch assortativity -- mean fraction of each probe's k=15 feature-space
  neighbours sharing its branch;
* concordance coherence g -- the same fraction per probe; low g marks a probe
  living among the other branch. On MGKDB the low-g survivors were the
  mixed-mode worklist. Here they should sit on the branch transition, which is
  checked directly against the transition mask;
* segmentation -- HDBSCAN in feature space with a bootstrap adjusted-Rand
  stability gate, compared with the true branch by ARI.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.cluster import HDBSCAN
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from scripts.generate_ionut_data import values
from resolution.noise import competition

CASE = "ionut-itg-tem-argmax-gamma"
INPUTS = ("R/L_Ti", "R/L_Te", "R/L_n", "ν", "β", "k_y scale")
BRANCH_NAMES = ("ITG", "TEM")
K = 15
TRANSITION = .05                 # branch-gap fraction, as in the benchmark mask
PLACEMENT_ARMS = ("space-filling", "gpr-var", "gpr-u50-g50", "gpr-grad", "vwrs", "vurs")


def transition_mask(G, scale):
    """Live branch competition: branches within 5% of the range AND growing.

    A gap test alone also admits regions where both branches are near zero --
    ITG stable and KBM below onset -- which is not mode competition. For
    ITG-KBM that dead zone is 93% of a gap-only band and it drags the
    concordance AUC from 0.968 to 0.554; for ITG-TEM it is 21% and changes
    almost nothing. The gap-only mask is still reported for comparison.
    """
    gap = np.abs(G[:, 0]-G[:, 1])/scale
    return (gap <= TRANSITION) & (G.max(axis=1) > TRANSITION*scale), gap <= TRANSITION


def probe_cloud(count, seed=4242):
    return qmc.Sobol(6, scramble=True, seed=seed).random_base2(int(np.log2(count)))


# Feature views, after the MGKDB study's circularity control. The branch label
# is argmax of the branch growth rates, so any view containing them makes branch
# coherence true by construction: measured assortativity 1.000 and a transition
# AUC of exactly 0.5. "branches" is kept only to show that leak.
VIEWS = {"observables": ("gamma", "omega"), "branches": ("gamma", "omega", "G0", "G1")}


def response_block(case, x, view="observables"):
    v = values(case, x)
    columns = dict(gamma=v["gamma"], omega=v["omega"], G0=v["G"][:, 0], G1=v["G"][:, 1])
    return v, np.column_stack([columns[name] for name in VIEWS[view]])


def features(x, block, stats, weight):
    """Scaled inputs beside the standardized response, weighted by w."""
    mean, sd = stats
    return np.hstack([x, weight*(block-mean)/sd])


def coherence(feature, labels, k=K):
    index = NearestNeighbors(n_neighbors=k+1).fit(feature).kneighbors(feature)[1][:, 1:]
    return (labels[index] == labels[:, None]).mean(axis=1)


def stable_segments(feature, seed=0, rounds=10, fraction=.8, min_cluster=60):
    """HDBSCAN segments plus mean bootstrap ARI on the overlapping points."""
    base = HDBSCAN(min_cluster_size=min_cluster, copy=True).fit_predict(feature)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(rounds):
        take = rng.choice(len(feature), int(fraction*len(feature)), replace=False)
        again = HDBSCAN(min_cluster_size=min_cluster, copy=True).fit_predict(feature[take])
        scores.append(adjusted_rand_score(base[take], again))
    return base, float(np.mean(scores)), float(np.std(scores))


def embed(feature, method, seed=0):
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit("umap-learn is not installed in this environment") from exc
        return umap.UMAP(n_neighbors=K, min_dist=.1, random_state=seed).fit_transform(feature)
    if method == "pacmap":
        try:
            import pacmap
        except ImportError as exc:
            raise SystemExit("pacmap is not installed in this environment") from exc
        return pacmap.PaCMAP(n_components=2, n_neighbors=K, random_state=seed).fit_transform(feature)
    raise ValueError(method)


def designs(run, case, seed=0):
    payload = json.loads((pathlib.Path(run)/"results.json").read_text(encoding="utf-8"))
    out = {}
    for row in payload["rows"]:
        if (row["status"] == "ok" and row["case"] == case and row["seed"] == seed
                and row["n"] == row["budget"] and row.get("x") is not None):
            out[row["arm"]] = np.asarray(row["x"])
    return out


def scatter(axis, xy, color, size, title, cmap=None, norm=None, **kw):
    order = np.argsort(size)
    image = axis.scatter(xy[order, 0], xy[order, 1], c=color[order] if np.ndim(color) else color,
                         s=size[order], cmap=cmap, norm=norm, linewidths=0, **kw)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([]); axis.set_yticks([])
    return image



# The two views the first sweep said to show side by side. Parameter space
# locates where the modes compete (low coherence finds the transition band);
# observable space separates the modes into islands, via the sign of omega.
DISPLAY = (("parameter space", "observables", 0.), ("observable space", "observables", 1.))


def sweep_views(x, weights):
    """Every view and weight, all measured in feature space."""
    rows = []
    for view in VIEWS:
        truth, block = response_block(CASE, x, view)
        stats = (block.mean(axis=0), block.std(axis=0))
        branch = np.argmax(truth["G"], axis=1)
        transition, gap_only = transition_mask(truth["G"], float(np.ptp(truth["G"])))
        for w in weights:
            if view != "observables" and w == 0:
                continue                      # identical to observables at w = 0
            f = features(x, block, stats, w)
            g = coherence(f, branch)
            seg, stab, stab_sd = stable_segments(f)
            rows.append(dict(view=view if w else "inputs only", weight=w,
                             circular=view == "branches",
                             assortativity=float(g.mean()),
                             segments=int(len(set(seg)) - (1 if -1 in seg else 0)),
                             noise_fraction=float((seg == -1).mean()),
                             segment_branch_ari=float(adjusted_rand_score(branch, seg)),
                             stability_ari=stab, stability_sd=stab_sd,
                             transition_auc=float(roc_auc_score(transition, 1-g)),
                             transition_auc_gap_only=float(roc_auc_score(gap_only, 1-g))))
            if rows[-1]["segments"] == 0:
                # All-unassigned agrees with itself trivially; not a stability.
                rows[-1]["stability_ari"] = rows[-1]["stability_sd"] = None
            stability = ("n/a" if rows[-1]["stability_ari"] is None
                         else f"{rows[-1]['stability_ari']:.2f}")
            print("{view:>16} w={weight:<4} assort {assortativity:.3f}  segments {segments}  "
                  "ARI {segment_branch_ari:.3f}  ".format(**rows[-1])
                  + f"stability {stability}  transition AUC {rows[-1]['transition_auc']:.3f}",
                  flush=True)
    return rows


def summarize(root):
    """One table and one heatmap over every case folder under `root`.

    Placement is expressed as a lift: an arm's transition-band share divided by
    the probe cloud's, so 1 means no targeting whatever the case's band size.
    """
    root = pathlib.Path(root)
    cases = []
    for path in sorted(root.glob("*/metrics.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    if not cases:
        raise SystemExit(f"no per-case metrics under {root}")
    arms = sorted({a for c in cases for a in c["placement"]})
    names = [c["case"].replace("ionut-", "") for c in cases]
    lift = np.full((len(arms), len(cases)), np.nan)
    for j, c in enumerate(cases):
        for i, arm in enumerate(arms):
            runs = c["placement"].get(arm, [])
            if runs:
                lift[i, j] = np.mean([r["transition_share"] for r in runs])/c["transition_fraction"]

    def row(c, view, weight):
        return next(r for r in c["sweep"] if r["view"] == view and r["weight"] == weight)

    figure, axes = plt.subplots(1, 2, figsize=(19, 7.5), layout="constrained",
                                gridspec_kw=dict(width_ratios=(1.25, 1)))
    image = axes[0].imshow(lift, cmap="RdBu_r", norm=matplotlib.colors.TwoSlopeNorm(1., .2, 2.5),
                           aspect="auto")
    for i in range(len(arms)):
        for j in range(len(cases)):
            if np.isfinite(lift[i, j]):
                axes[0].text(j, i, f"{lift[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[0].set_xticks(range(len(cases)), names, rotation=35, ha="right")
    axes[0].set_yticks(range(len(arms)), arms)
    axes[0].set_title("transition targeting: arm's band share ÷ probe share, mean of seeds\n"
                      "(1 = uniform, >1 targets the transition, <1 avoids it)")
    figure.colorbar(image, ax=axes[0], shrink=.8)

    metrics = [("parameter-space transition AUC", lambda c: row(c, "inputs only", 0.)["transition_auc"]),
               ("observable-space branch ARI", lambda c: row(c, "observables", 1.)["segment_branch_ari"]),
               ("observable-space stability", lambda c: row(c, "observables", 1.)["stability_ari"] or 0.),
               ("probe share in band", lambda c: c["transition_fraction"])]
    width = .8/len(metrics)
    for k, (label, get) in enumerate(metrics):
        axes[1].bar(np.arange(len(cases)) + (k-(len(metrics)-1)/2)*width,
                    [get(c) for c in cases], width, label=label)
    axes[1].set_xticks(range(len(cases)), names, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(.5, color="#999999", linewidth=.8, linestyle=":")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[1].set_title("embedding tooling per case (feature space)")
    figure.suptitle("Native 6D Ionut proxies — embedding and placement across all cases", fontsize=13)
    figure.savefig(root/"summary.png", dpi=110)
    plt.close(figure)

    summary = dict(cases=names, arms=arms, transition_lift=lift.tolist(),
                   per_case=[dict(case=c["case"], branches=c["branches"],
                                  transition_fraction=c["transition_fraction"],
                                  parameter_auc=row(c, "inputs only", 0.)["transition_auc"],
                                  observable_ari=row(c, "observables", 1.)["segment_branch_ari"],
                                  observable_stability=row(c, "observables", 1.)["stability_ari"],
                                  trustworthiness={f"{d['label']}/{d['method']}": d["trustworthiness"]
                                                   for d in c["display"]})
                             for c in cases])
    (root/"summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"summarized {len(cases)} cases into {root/'summary.png'}")


def main():
    global CASE, BRANCH_NAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--probes", type=int, default=8192)
    parser.add_argument("--weights", type=float, nargs="+", default=[0., .25, .5, 1., 2.])
    parser.add_argument("--run", type=pathlib.Path,
                        default=pathlib.Path("results/6d-ionut-spine-2026-09-20"))
    parser.add_argument("--case", default=CASE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--summarize", action="store_true",
                        help="combine per-case metrics under --output into one summary")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    CASE = args.case
    BRANCH_NAMES = ("ITG", "KBM") if "itg-kbm" in CASE else ("ITG", "TEM")
    figures = args.output/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    x = probe_cloud(args.probes)
    truth, block = response_block(CASE, x, "observables")
    stats = (block.mean(axis=0), block.std(axis=0))
    branch = np.argmax(truth["G"], axis=1)
    scale = float(np.ptp(truth["G"]))
    transition, gap_only = transition_mask(truth["G"], scale)
    contest = competition(truth["G"])
    gamma, omega = truth["gamma"], truth["omega"]
    size = 2 + 40*(gamma-gamma.min())/max(float(np.ptp(gamma)), 1e-12)
    branch_colors = np.array(["#3b6fd6", "#e0702b"])

    sweep = sweep_views(x, args.weights)

    # Embed each display view once, with both projections.
    shown = []
    sub = np.random.default_rng(1).choice(len(x), min(3000, len(x)), replace=False)
    for label, view, w in DISPLAY:
        f = features(x, block, stats, w)
        g = coherence(f, branch)
        seg, stab, _ = stable_segments(f)
        for method in ("pacmap", "umap"):
            xy = embed(f, method)
            trust = float(trustworthiness(f[sub], xy[sub], n_neighbors=K))
            print(f"{label} / {method}: trustworthiness {trust:.3f}", flush=True)
            shown.append(dict(label=label, view=view, weight=w, method=method, xy=xy,
                              trust=trust, feature=f, coherence=g, segments=seg, stability=stab))

    # Sheet 1: overview. One row per view x projection.
    figure, axes = plt.subplots(len(shown), 4, figsize=(20, 4.7*len(shown)), layout="constrained")
    for i, item in enumerate(shown):
        xy = item["xy"]
        head = f"{item['label']} — {item['method'].upper()} (trust {item['trust']:.3f})"
        scatter(axes[i, 0], xy, branch_colors[branch], size, f"{head}\nbranch, size = γ", alpha=.8)
        image = scatter(axes[i, 1], xy, contest, size, "branch competition s(x)",
                        cmap="YlOrRd", alpha=.85)
        figure.colorbar(image, ax=axes[i, 1], shrink=.75)
        image = scatter(axes[i, 2], xy, omega, size, "frequency ω", cmap="coolwarm", alpha=.85)
        figure.colorbar(image, ax=axes[i, 2], shrink=.75)
        seg = item["segments"]
        colors = np.where(seg[:, None] == -1, np.array([[.82, .82, .82, 1.]]),
                          plt.get_cmap("tab10")(np.mod(seg, 10)))
        count = len(set(seg)) - (1 if -1 in seg else 0)
        # With no segments every bootstrap agrees trivially (all unassigned),
        # so a stability number there would be meaningless.
        scatter(axes[i, 3], xy, colors, size,
                (f"HDBSCAN in feature space: {count} segments, stability {item['stability']:.2f}"
                 if count else "HDBSCAN in feature space: no segments (stability n/a)\n"
                               "grey = unassigned: a uniform box has no density islands"),
                alpha=.85)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=n)
               for c, n in zip(branch_colors, BRANCH_NAMES)]
    axes[0, 0].legend(handles=handles, loc="lower right", fontsize=8)
    figure.suptitle(f"{CASE}, {len(x)} fixed probes.  Parameter space = scaled inputs only; "
                    "observable space = inputs + 1 × standardized (γ, ω).  "
                    "Neighbourhoods and segments computed in feature space; projections are display only.",
                    fontsize=12)
    figure.savefig(figures/"01-embedding-overview.png", dpi=100)
    plt.close(figure)

    # Sheet 2: concordance against the true transition band, in both views.
    figure, axes = plt.subplots(2, 3, figsize=(17, 11), layout="constrained")
    bins = np.linspace(0, 1, 21)
    for i, item in enumerate(s for s in shown if s["method"] == "pacmap"):
        g, xy = item["coherence"], item["xy"]
        auc = roc_auc_score(transition, 1-g)
        image = scatter(axes[i, 0], xy, g, size,
                        f"{item['label']}: coherence g (own-branch share of {K} neighbours)",
                        cmap="viridis_r", alpha=.85)
        figure.colorbar(image, ax=axes[i, 0], shrink=.8)
        scatter(axes[i, 1], xy, np.where(transition, "#d62728", "#cfcfcf"), size,
                f"live transition: gap ≤ {TRANSITION:.0%} of range and a branch growing", alpha=.85)
        axes[i, 2].hist(g[~transition], bins=bins, alpha=.6, density=True, label="away from transition")
        axes[i, 2].hist(g[transition], bins=bins, alpha=.6, density=True, label="in transition band")
        axes[i, 2].set_xlabel("coherence g"); axes[i, 2].set_ylabel("density")
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f"low coherence as a transition detector: AUC {auc:.3f}")
    figure.suptitle(f"{CASE}: concordance lens on a known answer (PaCMAP display).  Top: parameter "
                    "space.  Bottom: observable space.\nLow coherence can only mark the transition where "
                    "the two branches actually interpenetrate in that view.", fontsize=12)
    figure.savefig(figures/"02-concordance-vs-transition.png", dpi=100)
    plt.close(figure)

    # Sheet 3: placement. Each arm's final design, mapped to its nearest probe
    # in feature space, drawn on the fixed embeddings.
    # Placement statistics over every arm and seed, measured in physical space
    # from the designs' own branch gaps. The figure draws seed 0 only.
    placement = {}
    for seed in args.seeds:
        for arm, d in sorted(designs(args.run, CASE, seed).items()):
            dtruth = values(CASE, d)
            live, _ = transition_mask(dtruth["G"], scale)
            placement.setdefault(arm, []).append(dict(
                seed=seed, points=int(len(d)), transition_share=float(live.mean()),
                second_branch_share=float((np.argmax(dtruth["G"], axis=1) == 1).mean())))
    found = designs(args.run, CASE, args.seeds[0])
    arms = [a for a in PLACEMENT_ARMS if a in found]
    if arms:
        panels = [s for s in shown if s["method"] == "pacmap"]
        figure, axes = plt.subplots(len(panels), len(arms), figsize=(3.7*len(arms), 4.2*len(panels)),
                                    squeeze=False, layout="constrained")
        for j, arm in enumerate(arms):
            d = found[arm]
            dtruth, dblock = response_block(CASE, d, "observables")
            share = float(transition_mask(dtruth["G"], scale)[0].mean())
            for i, item in enumerate(panels):
                probe = NearestNeighbors(n_neighbors=1).fit(item["feature"]).kneighbors(
                    features(d, dblock, stats, item["weight"]), return_distance=False)[:, 0]
                xy = item["xy"]
                axis = axes[i, j]
                axis.scatter(xy[:, 0], xy[:, 1], c=branch_colors[branch], s=2, alpha=.12, linewidths=0)
                axis.scatter(xy[probe, 0], xy[probe, 1], c=np.arange(len(probe)), cmap="plasma",
                             s=10, edgecolors="black", linewidths=.2)
                axis.set_xticks([]); axis.set_yticks([])
                if i == 0:
                    axis.set_title(f"{arm}\n{share:.0%} in transition band  "
                                   f"(probes {transition.mean():.0%})", fontsize=9)
                if j == 0:
                    axis.set_ylabel(item["label"])
        figure.suptitle(f"{CASE}: where each method's N={len(d)} design lands (seed {args.seeds[0]}, "
                        "PaCMAP).  Points coloured by acquisition order; faint background by true "
                        f"branch (blue {BRANCH_NAMES[0]}, orange {BRANCH_NAMES[1]}).", fontsize=12)
        figure.savefig(figures/"03-placement-on-embedding.png", dpi=100)
        plt.close(figure)

    metrics = dict(case=CASE, branches=list(BRANCH_NAMES), probes=len(x), k=K,
                   transition_fraction=float(transition.mean()),
                   gap_only_fraction=float(gap_only.mean()),
                   second_branch_fraction=float(branch.mean()), sweep=sweep,
                   display=[dict(label=s["label"], method=s["method"], weight=s["weight"],
                                 trustworthiness=s["trust"], stability=s["stability"])
                            for s in shown],
                   placement=placement, seconds=time.perf_counter()-started)
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"done in {metrics['seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
