"""Per-motif summaries: what each motif is, and how much of the system it holds.

These are the numbers a motif table is built from -- mean properties, population,
and population against time. They are deliberately plain functions over arrays so
that a caller can tabulate, plot or export them without this module deciding.

TWO CONVENTIONS THAT MATTER
---------------------------
**Population is a share of ATOMS, not of motifs.** A vocabulary of 18 motifs
where one holds 40% of atoms is not "18 equal things"; quoting the motif count
without the populations implies a uniformity that is not there.

**Ring size is reported as a mean with a spread, not a mode.** The dominant ring
size is a single integer and hides how mixed an environment is: a motif that is
41% three-rings and 18% six-rings has a dominant size of 3 and a mean near 4.4,
and only the second is honest about it. `ring_size_stats()` returns both the mean
and the s.d. of the distribution -- the s.d. is the spread of ring sizes the
motif participates in, NOT a standard error on the mean. At typical n the
standard error is ~0.001 and would imply a precision the quantity does not have.
"""
from __future__ import annotations

import numpy as np

__all__ = ["motif_table", "population_over_time", "ring_size_stats"]


def motif_table(labels, properties, ranks=None, min_percent=0.0):
    """One row per motif: population and the mean of each named property.

    labels     : (N,) motif label per atom
    properties : {name: (N,) array}
    ranks      : optional {label: motif_relabel}, added as a column
    min_percent: drop motifs holding less than this share of atoms

    Returns a list of dicts, largest population first. NaNs are ignored
    per-property rather than per-row, so a property that is undefined for a
    system (no per-atom energy, say) leaves that one column empty instead of
    discarding the motif.
    """
    labels = np.asarray(labels)
    n = len(labels)
    rows = []
    for k in np.unique(labels):
        m = labels == k
        cnt = int(m.sum())
        pct = 100.0 * cnt / n
        if pct < min_percent:
            continue
        row = {"motif": int(k), "n_atoms": cnt, "percent_of_atoms": round(pct, 3)}
        if ranks is not None:
            row["motif_relabel"] = int(ranks.get(int(k), 0))
        for name, vals in properties.items():
            v = np.asarray(vals, dtype=float)[m]
            v = v[np.isfinite(v)]
            row[name] = round(float(v.mean()), 4) if v.size else None
        rows.append(row)
    return sorted(rows, key=lambda r: -r["percent_of_atoms"])


def population_over_time(labels, n_atoms, motifs=None, times=None):
    """Population (%) of each motif in each frame.

    labels  : (n_frames * n_atoms,) in frame-major order -- the order the
              featuriser emits, so frame f occupies [f*n_atoms, (f+1)*n_atoms).
    times   : optional (n_frames,) of simulation times, carried through as-is.

    Returns {"times": [...], "motifs": [...], "percent": {motif: [...]}}.

    The frame-major assumption is the thing to check if a series looks wrong: if
    the labels were written per-atom-then-per-frame the series will still plot,
    it will just be meaningless.
    """
    labels = np.asarray(labels)
    if len(labels) % n_atoms:
        raise ValueError(f"{len(labels)} labels is not a whole number of "
                         f"{n_atoms}-atom frames")
    nf = len(labels) // n_atoms
    byframe = labels.reshape(nf, n_atoms)
    if motifs is None:
        motifs = [int(k) for k in np.unique(labels)]
    pct = {int(k): [float(100.0 * (fr == k).mean()) for fr in byframe] for k in motifs}
    return {"times": list(times) if times is not None else list(range(nf)),
            "motifs": [int(k) for k in motifs], "percent": pct}


def ring_size_stats(ring_counts):
    """Mean ring size and its spread, from {size: count} per motif.

    Returns {motif: (mean, sd, total)}. See the module docstring on why the mean
    is reported rather than the dominant size, and on what the sd is.
    """
    out = {}
    for motif, counts in ring_counts.items():
        sizes = np.array(sorted(counts), dtype=float)
        w = np.array([counts[int(s)] for s in sizes], dtype=float)
        tot = w.sum()
        if tot <= 0:
            out[int(motif)] = (float("nan"), float("nan"), 0)
            continue
        mu = float((sizes * w).sum() / tot)
        sd = float(np.sqrt((w * (sizes - mu) ** 2).sum() / tot))
        out[int(motif)] = (round(mu, 3), round(sd, 3), int(tot))
    return out
