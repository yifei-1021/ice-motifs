"""Figures for a motif analysis, coloured by motif_relabel.

COLOUR IS NOT A FREE CHOICE HERE
--------------------------------
Every figure in a set must give the same motif the same colour, or panels stop
being comparable and a reader will draw conclusions from a colour that only
reflects which figure they are looking at. So colour is driven by ONE table --
the relabel ranking -- passed into every function, rather than each plot picking
its own.

The ramp is applied to the RANK over the full vocabulary 1..n, never to the
subset present in the data being plotted. A single frame contains fewer motifs
than the vocabulary; ranging on what is present rescales the ramp per frame and
the colours drift while every frame looks individually fine.

`motif_colors()` takes `n_motifs` explicitly for that reason, and will not infer
it from the labels it is given.
"""
from __future__ import annotations

import numpy as np

__all__ = ["motif_colors", "motif_map", "property_panels",
           "population_vs_time", "distributions"]

_LABEL = {"rho": r"$\rho_\mathrm{voro}$ (g cm$^{-3}$)",
          "A": "tetrahedrality $A$", "q6": r"$\overline{q}_6$",
          "energy": "potential energy (eV/atom)", "time": "time (ns)"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def motif_colors(ranks, n_motifs, cmap="turbo"):
    """{motif: rgba} from the relabel rank, over the FULL vocabulary.

    `n_motifs` is the size of the vocabulary, not of `ranks` -- pass the number
    the clustering produced even when plotting a subset, or the two figures will
    disagree about what rank 5 looks like.
    """
    plt = _plt()
    cm = plt.get_cmap(cmap)
    if n_motifs < len(ranks):
        raise ValueError(f"n_motifs={n_motifs} is smaller than the {len(ranks)} "
                         "ranks given; pass the full vocabulary size")
    den = max(n_motifs - 1, 1)
    return {int(k): cm((int(r) - 1) / den) for k, r in ranks.items()}


def motif_map(embedding, labels, ranks, n_motifs, out, *, title=None,
              subsample=200000, seed=0, point_size=0.6):
    """The 2-D projection, each atom coloured by its motif."""
    plt = _plt()
    emb = np.asarray(embedding); lab = np.asarray(labels)
    if subsample and len(emb) > subsample:
        idx = np.random.default_rng(seed).choice(len(emb), subsample, replace=False)
        emb, lab = emb[idx], lab[idx]
    col = motif_colors(ranks, n_motifs)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    for k in sorted(col, key=lambda k: ranks.get(k, 0)):
        m = lab == k
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=point_size, lw=0,
                       color=col[k], rasterized=True)
    ax.set_xlabel("projection axis 1"); ax.set_ylabel("projection axis 2")
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    fig.savefig(out); plt.close(fig)
    return out


def property_panels(embedding, properties, out, *, subsample=200000, seed=0,
                    point_size=0.6, clim=None):
    """One panel per property, the projection coloured by that property."""
    plt = _plt()
    emb = np.asarray(embedding)
    names = list(properties)
    if subsample and len(emb) > subsample:
        idx = np.random.default_rng(seed).choice(len(emb), subsample, replace=False)
        emb = emb[idx]
        properties = {k: np.asarray(v)[idx] for k, v in properties.items()}
    nc = min(3, len(names)); nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.4 * nr),
                             constrained_layout=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        v = np.asarray(properties[name], dtype=float)
        ok = np.isfinite(v)
        lo, hi = (clim or {}).get(name, (None, None))
        sc = ax.scatter(emb[ok, 0], emb[ok, 1], c=v[ok], s=point_size, lw=0,
                        cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
        fig.colorbar(sc, ax=ax, shrink=0.85)
        ax.set_title(_LABEL.get(name, name), fontsize=10, loc="left")
        ax.set_xlabel("axis 1"); ax.set_ylabel("axis 2")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.savefig(out); plt.close(fig)
    return out


def population_vs_time(series, ranks, n_motifs, out, *, title=None,
                       min_percent=2.0, xlabel="time (ns)"):
    """Motif populations against time, from analysis.population_over_time()."""
    plt = _plt()
    col = motif_colors(ranks, n_motifs)
    t = series["times"]
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    drawn = 0
    for k in sorted(series["percent"], key=lambda k: ranks.get(k, 0)):
        y = series["percent"][k]
        if max(y) < min_percent:        # below this it is noise, not a population
            continue
        ax.plot(t, y, lw=1.1, color=col.get(k, "0.6"), label=str(k))
        drawn += 1
    ax.set_xlabel(xlabel); ax.set_ylabel("population (%)")
    if drawn:
        ax.legend(title="motif", frameon=False, fontsize=8, ncol=2,
                  loc="center left", bbox_to_anchor=(1.0, 0.5))
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    fig.savefig(out); plt.close(fig)
    return out


def distributions(properties, labels, ranks, n_motifs, out, *,
                  motifs=None, bins=25, title=None):
    """Per-motif distribution of each property, one panel each.

    `motifs` selects which to draw; everything else is pooled into one grey
    curve so the panel still sums to the whole system rather than silently
    dropping the remainder.
    """
    plt = _plt()
    lab = np.asarray(labels)
    col = motif_colors(ranks, n_motifs)
    keep = list(motifs) if motifs is not None else sorted(col, key=lambda k: ranks.get(k, 0))
    names = list(properties)
    nc = min(2, len(names)); nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.2 * nr),
                             constrained_layout=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        v = np.asarray(properties[name], dtype=float)
        ok = np.isfinite(v)
        lo, hi = np.percentile(v[ok], [0.5, 99.5])
        edges = np.linspace(lo, hi, bins + 1)
        rest = ok & ~np.isin(lab, keep)
        if rest.any():
            h, _ = np.histogram(v[rest], bins=edges, density=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), h, lw=1.0, color="0.6",
                    label="other motifs")
        for k in keep:
            m = ok & (lab == k)
            if m.sum() < 10:
                continue
            h, _ = np.histogram(v[m], bins=edges, density=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), h, lw=1.2, color=col.get(k, "0.6"),
                    label=str(k))
        ax.set_xlabel(_LABEL.get(name, name)); ax.set_ylabel("density")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, title="motif")
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.savefig(out); plt.close(fig)
    return out
