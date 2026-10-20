"""Rank motifs along one projection axis, for colouring.

TWO NUMBERINGS, AND WHY BOTH EXIST
----------------------------------
`motif` is the vocabulary: the identifier every table, CSV and figure refers to.
It carries no order -- motif 7 is not "between" 6 and 8 in any sense.

`motif_relabel` is a RANK along one axis of one projection: rank 1 is the motif
whose median lies at the low end, rank n at the high end. It is useless as an
identifier and must never replace `motif`, but it is the only thing a continuous
colour ramp can be driven by. Renderers that colour by a scalar (OVITO's Rainbow,
matplotlib's jet) need a quantity whose ORDER means something; the vocabulary
number does not qualify.

RANK, NOT COORDINATE
--------------------
The ramp encodes order along the axis, not distance along it. Where many motifs
sit close together, a linear map of the coordinate would give most of them the
same shade; ranking spreads them evenly, which is what makes the figure readable.
The cost is that equal colour steps do not mean equal distance, and the caption
should say so.

WHICH AXIS IS NOT A DEFAULT
---------------------------
The informative axis is a property of the map, not a convention: it is whichever
one the physics runs along, and it differs between systems. Choose it by
measuring, not by assuming -- `axis_by_correlation()` picks the axis whose
correlation with a per-motif property is larger. Whatever you pick must match the
axis any companion figure was coloured by, or render and figure will disagree
while both look correct.

ONE RANKING PER CLUSTERING
--------------------------
Systems that share a clustering must share a ranking. Ranking each subset on its
own rows gives the same motif a different colour in each, and the panels silently
stop being comparable.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rank_motifs", "axis_by_correlation", "relabel_map"]


def rank_motifs(embedding, labels, axis=0, reverse=False):
    """Map each motif label to its rank along `axis` of `embedding`.

    embedding : (N, >=2) array of projected coordinates
    labels    : (N,) integer motif labels, any numbering
    axis      : 0 for the first projection axis, 1 for the second
    reverse   : rank descending, so the low end gets the HIGH rank. Use this when
                the companion figure used a reversed colour ramp.

    Returns {label: rank}, ranks starting at 1, and the per-label medians.
    Ranking is on the MEDIAN rather than the mean: a motif straddling a sparse
    region has a mean pulled into territory it does not occupy.
    """
    embedding = np.asarray(embedding)
    labels = np.asarray(labels)
    if embedding.ndim != 2 or embedding.shape[1] <= axis:
        raise ValueError(f"embedding {embedding.shape} has no axis {axis}")
    if len(labels) != len(embedding):
        raise ValueError(f"{len(labels)} labels for {len(embedding)} rows")

    uniq = np.unique(labels)
    med = {int(k): float(np.median(embedding[labels == k, axis])) for k in uniq}
    order = sorted(med, key=med.get, reverse=bool(reverse))
    return {k: i + 1 for i, k in enumerate(order)}, med


def axis_by_correlation(embedding, labels, values):
    """Pick the projection axis that carries a per-row property, by |Pearson r|.

    `values` is one number per row -- a tetrahedrality, a density, anything the
    ordering should follow. Returns (axis, r_axis0, r_axis1).

    This is the measurement that replaces guessing. Report both correlations, not
    just the winner: if they are close the axis choice is arbitrary and the
    figure should not lean on it.
    """
    embedding = np.asarray(embedding)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    r = []
    for ax in (0, 1):
        c = np.corrcoef(embedding[ok, ax], values[ok])[0, 1]
        r.append(0.0 if not np.isfinite(c) else float(c))
    return (0 if abs(r[0]) >= abs(r[1]) else 1), r[0], r[1]


def relabel_map(embedding, labels, axis=0, reverse=False):
    """rank_motifs(), plus the population of each motif, ready to write as a legend."""
    rank, med = rank_motifs(embedding, labels, axis=axis, reverse=reverse)
    labels = np.asarray(labels)
    n = len(labels)
    rows = []
    for k in sorted(rank, key=rank.get):
        cnt = int((labels == k).sum())
        rows.append({"motif": int(k), "motif_relabel": rank[k],
                     "median_axis": round(med[k], 4),
                     "n_atoms": cnt,
                     "percent_of_atoms": round(100.0 * cnt / n, 4)})
    return rows
