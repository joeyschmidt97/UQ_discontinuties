"""
Eyeball the test manifold before trusting any benchmark run on it.

Three figures:
  fig_manifold_slices.png  -- gamma, omega, dominant branch and the
                              competing-mode band on a 2D slice
  fig_manifold_sweeps.png  -- 1D sweeps showing the kink (gamma), the jump
                              (omega) and the band gap (alignment)
  fig_manifold_stats.png   -- branch occupancy and the band's volume fraction,
                              i.e. is the benchmark actually a competition

Run:  python plot_manifold.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold import BRANCH_NAMES, CAMPAIGN_AXES, Manifold

OUT = pathlib.Path(__file__).parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DIM = len(CAMPAIGN_AXES)
NAMES = [a.name for a in CAMPAIGN_AXES]


def _slice(ix, iy, n=260, fixed=0.5):
    gx, gy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    U = np.full((gx.size, DIM), fixed)
    U[:, ix] = gx.ravel()
    U[:, iy] = gy.ravel()
    return gx, gy, U


def fig_slices():
    # Te_ped (col 0) vs w_Te (col 2): the two axes that move beta, the gradient
    # AND the rational-surface count, so all three features show up at once.
    ix, iy = 0, 2
    gx, gy, U = _slice(ix, iy)
    ax0, ax1 = CAMPAIGN_AXES[ix], CAMPAIGN_AXES[iy]
    X, Y = ax0.to_phys(gx), ax1.to_phys(gy)

    cases = [("softmax T=0.05, smooth gate", dict(mode="softmax", T=0.05, align="smooth")),
             ("argmax, smooth gate", dict(mode="argmax", align="smooth")),
             ("argmax + band gap", dict(mode="argmax", align="gap"))]

    fig, axes = plt.subplots(3, len(cases), figsize=(4.1 * len(cases), 10.5))
    for j, (title, kw) in enumerate(cases):
        ev = Manifold(**kw).evaluate(U)
        for i, (key, cmap, lab) in enumerate([
                ("gamma", "viridis", r"$\gamma$"),
                ("omega", "coolwarm", r"$\omega_r$  (sign = drift direction)"),
                ("mix", "magma", "competing-mode indicator")]):
            a = axes[i, j]
            Z = ev[key].reshape(gx.shape)
            kwargs = dict(shading="auto", cmap=cmap)
            if key == "omega":
                v = np.abs(Z).max()
                kwargs.update(vmin=-v, vmax=v)
            pc = a.pcolormesh(X, Y, Z, **kwargs)
            fig.colorbar(pc, ax=a, label=lab if j == 0 else "")
            if i == 0:
                a.set_title(title, fontsize=9)
            a.set_xlabel(NAMES[ix]); a.set_ylabel(NAMES[iy])
    fig.suptitle("Manifold slice: pedestal-top $T_e$ vs $T_e$ width "
                 "(other axes at box centre)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_manifold_slices.png", dpi=140)
    plt.close(fig)


def fig_sweeps():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    t = np.linspace(0, 1, 900)

    for col, (ix, label) in enumerate([(0, "Te_ped_scale"), (2, "w_Te_scale"),
                                       (5, "ky_scale")]):
        U = np.full((len(t), DIM), 0.5)
        U[:, ix] = t
        xs = CAMPAIGN_AXES[ix].to_phys(t)

        hard = Manifold(mode="argmax", align="smooth").evaluate(U)
        soft = Manifold(mode="softmax", T=0.08, align="smooth").evaluate(U)
        gap = Manifold(mode="argmax", align="gap").evaluate(U)

        a = axes[0, col]
        for b in range(len(BRANCH_NAMES)):
            a.plot(xs, hard["G"][b], ":", lw=1.1, label=BRANCH_NAMES[b])
        a.plot(xs, hard["gamma"], "-", color="0.55", lw=1.7, label="argmax (kink)")
        a.plot(xs, soft["gamma"], "-", color="k", lw=2.2, label="softmax (smooth)")
        a.plot(xs, gap["gamma"], "-", color="#d95f02", lw=1.5, label="argmax + gap (jump)")
        a.set_ylabel(r"$\gamma$ [$c_s/a$]"); a.set_title(label, fontsize=9)
        if col == 0:
            a.legend(frameon=False, fontsize=7, ncol=2)

        a = axes[1, col]
        a.axhline(0, color="0.7", lw=0.8)
        a.plot(xs, hard["omega"], "-", color="0.55", lw=1.7, label="argmax (jump)")
        a.plot(xs, soft["omega"], "-", color="k", lw=2.2, label="softmax")
        a.set_xlabel(label); a.set_ylabel(r"$\omega_r$ [$c_s/a$]")
        if col == 0:
            a.legend(frameon=False, fontsize=7)

    for a in axes.ravel():
        a.spines[["right", "top"]].set_visible(False)
    fig.suptitle(r"$\gamma$ kinks where branches cross; $\omega_r$ jumps sign; "
                 "the alignment gate adds a true discontinuity", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_manifold_sweeps.png", dpi=140)
    plt.close(fig)


def fig_stats(n=40000):
    U = np.random.default_rng(7).random((n, DIM))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ev = Manifold(mode="argmax", align="smooth").evaluate(U)
    occ = np.bincount(ev["label"], minlength=len(BRANCH_NAMES)) / n
    axes[0].bar(BRANCH_NAMES, occ, color=[f"C{i}" for i in range(len(occ))])
    axes[0].set_ylabel("fraction of box where branch dominates")
    axes[0].set_title("Branch occupancy (is it a competition?)", fontsize=9)
    for i, v in enumerate(occ):
        axes[0].text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)

    axes[1].hist(ev["mix"], bins=60, color="#d95f02")
    axes[1].set_yscale("log")
    axes[1].axvline(0.30, color="k", ls="--", lw=1.2, label="band threshold 0.30")
    axes[1].set_xlabel("competing-mode indicator")
    axes[1].set_title(f"Band volume fraction = {np.mean(ev['mix'] >= 0.30):.3f}",
                      fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)

    gap = Manifold(mode="argmax", align="gap").evaluate(U)
    d = np.abs(gap["gamma"] - ev["gamma"])
    rel = d / max(ev["gamma"].max(), 1e-12)
    axes[2].hist(rel[rel > 1e-6], bins=60, color="#1b9e77")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$|\Delta\gamma|$ / max $\gamma$ from the band gap")
    axes[2].set_title(f"Gap bites on {np.mean(rel > 0.01):.3f} of the box, "
                      f"max jump {rel.max():.2f}", fontsize=9)

    for a in axes:
        a.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_manifold_stats.png", dpi=140)
    plt.close(fig)
    return occ, float(np.mean(ev["mix"] >= 0.30)), float(np.mean(rel > 0.01))


if __name__ == "__main__":
    fig_slices()
    fig_sweeps()
    occ, band, gap = fig_stats()
    print("branch occupancy :", dict(zip(BRANCH_NAMES, np.round(occ, 3))))
    print(f"competing-mode band volume fraction : {band:.3f}")
    print(f"band-gap footprint (>1% of max gamma): {gap:.3f}")
    print(f"figures -> {OUT}")
