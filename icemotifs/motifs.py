"""
Stage 5 - motif analysis.

Everything downstream of the 18-motif codebook: the ring network, the temporal
grammar (transition matrix and motif lifetimes), the spatial grammar (neighbour
co-occurrence), and per-motif property statistics.

This module replaces five loose scripts and three notebooks that each carried
their own copy of the same neighbour-list code:

    makeRNN.py            -> rnn_neighbour_table
    makeRings.py          -> ring_network / ring_cycles
    getStats.py           -> label_rings
    makeNgbrAve.py        -> neighbour_average
    makeNgbrClass.py      -> neighbour_class_counts
    over-time.ipynb       -> transition_matrix / motif_lifetimes
    network.ipynb         -> ring_fractions / cooccurrence_matrix

CUTOFF CONVENTION - read this before changing any number
--------------------------------------------------------
ASE's NeighborList takes a per-atom *radius*, and two atoms are neighbours when
their separation is less than the SUM of their two radii. So the original
`NeighborList([1.6] * n)` is a 3.2 A cutoff, not a 1.6 A one, and
`NeighborList([3.0] * n)` is a 6.0 A search radius. Both matched the manuscript;
the code just did not look like it. Every function here takes `cutoff` as the
real interatomic cutoff and halves it internally, so the numbers in the code now
read the same as the numbers in the paper.

    ring network          cutoff = 3.2 A   (first coordination shell)
    RNN neighbour table   cutoff = 6.0 A   (search radius, truncated to k=40)
    motif visualisation   cutoff = 6.5 A   (Methods; plot_hdbscan1.ipynb rc=3.25)
"""

from __future__ import annotations

import numpy as np


# ============================================================ neighbour lists

def _ase_neighbourlist(atoms, cutoff):
    """ASE NeighborList for a true interatomic `cutoff` (see module docstring)."""
    from ase.neighborlist import NeighborList
    nl = NeighborList([cutoff / 2.0] * len(atoms),
                      self_interaction=False, bothways=True)
    nl.update(atoms)
    return nl


def rnn_neighbour_table(atoms, cutoff=6.0, k=40):
    """Sorted k-nearest-neighbour index and distance tables.

    Returns (idx, dist), both (n_atoms, k) with float distances. Atoms with
    fewer than k neighbours inside `cutoff` are padded.

    PADDING CHANGED ON PURPOSE. The original padded with -1 and downstream code
    then did `df.iloc[neighbour_indices]`, where -1 silently selects the LAST
    atom of the frame. Every atom short of 40 neighbours therefore contributed
    the same arbitrary atom, once per missing slot, to its own neighbourhood
    average. Padding is now -1 in the index table but the consumers here
    (`neighbour_average`, `neighbour_class_counts`) mask it out explicitly.
    """
    nl = _ase_neighbourlist(atoms, cutoff)
    n = len(atoms)
    idx = np.full((n, k), -1, dtype=np.int64)
    dist = np.full((n, k), np.nan, dtype=np.float64)
    for i in range(n):
        j, _ = nl.get_neighbors(i)
        if len(j) == 0:
            continue
        d = np.array([atoms.get_distance(i, jj, mic=True) for jj in j])
        order = np.argsort(d)
        j, d = np.asarray(j)[order][:k], d[order][:k]
        idx[i, :len(j)] = j
        dist[i, :len(d)] = d
    return idx, dist


# ============================================================ rings

def ring_network(atoms, cutoff=3.2):
    """networkx graph of the O-O network within `cutoff`."""
    import networkx as nx
    nl = _ase_neighbourlist(atoms, cutoff)
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    for i in range(len(atoms)):
        for j in nl.get_neighbors(i)[0]:
            if i != j:
                g.add_edge(int(i), int(j))
    return g


def ring_cycles(atoms, cutoff=3.2, sizes=(3, 4, 5, 6, 7), method="cycle_basis"):
    """Rings of each size, as {size: (n_rings, size) int array}.

    method="cycle_basis" reproduces the published analysis exactly
    (networkx.cycle_basis, which NetworkX attributes to Paton 1969).

    ⚠  A cycle *basis* is not the set of primitive rings. It returns exactly
    |E| - |V| + 1 cycles, it is basis-dependent (the answer changes with node
    ordering), and it is not the King / Franzblau shortest-path ring set that
    the ring-statistics literature means by "primitive rings". The published
    figures use it, so it stays the default and stays reproducible - but the
    Methods should say "cycle basis" rather than "primitive ring cycles".

    method="minimum_cycle_basis" is the deterministic, shortest-total-length
    basis. It is a strictly better-defined object and is offered here so the
    sensitivity can be reported; it is NOT what the published figures used.

    Empty sizes return a well-formed (0, size) array. The original returned
    np.array([]) with shape (0,), which made `arr.shape[1]` raise downstream -
    the reason `getStats.py` crashed on frames with no heptamers.
    """
    import networkx as nx
    g = ring_network(atoms, cutoff=cutoff)
    if method == "cycle_basis":
        cycles = nx.cycle_basis(g)
    elif method == "minimum_cycle_basis":
        cycles = nx.minimum_cycle_basis(g)
    else:
        raise ValueError(f"unknown method {method!r}")
    out = {}
    for s in sizes:
        found = [sorted(c) for c in cycles if len(c) == s]
        out[s] = (np.asarray(found, dtype=np.int64) if found
                  else np.empty((0, s), dtype=np.int64))
    return out


def ring_fractions(ring_dict):
    """{size: count} and each size as a percentage of all rings counted."""
    counts = {s: int(a.shape[0]) for s, a in ring_dict.items()}
    total = sum(counts.values())
    frac = {s: (100.0 * c / total if total else 0.0) for s, c in counts.items()}
    return counts, frac


def label_rings(ring_dict, atom_labels):
    """Replace atom indices in each ring by their motif label.

    `atom_labels` is a 1-D array indexed by 0-based atom index. The original
    (`getStats.py`) looked the label up through a metadata DataFrame with
    `atom-index == i + 1`, i.e. 1-based, and raised IndexError whenever an atom
    was absent from the frame's metadata. Passing the array directly removes
    both the ambiguity and the failure mode.
    """
    labels = np.asarray(atom_labels)
    out = {}
    for s, arr in ring_dict.items():
        out[s] = (labels[arr] if arr.size else
                  np.empty(arr.shape, dtype=labels.dtype))
    return out


# ============================================================ temporal grammar

def transition_matrix(labels_by_time, n_motifs=18, one_based=True):
    """Unnormalised and row-normalised motif transition matrices.

    `labels_by_time` is (n_times, n_atoms): one row per frame, motif label per
    atom, frames in increasing time order and equally spaced. Counts every
    consecutive pair, for every atom.

    Returns (counts, probabilities). Rows with no observed transition are left
    as all-zero rather than being turned into NaN by the division.
    """
    a = np.asarray(labels_by_time, dtype=np.int64)
    if one_based:
        a = a - 1
    if a.min() < 0 or a.max() >= n_motifs:
        raise ValueError(f"labels out of range after 0-basing: "
                         f"[{a.min()}, {a.max()}] for n_motifs={n_motifs}")
    counts = np.zeros((n_motifs, n_motifs), dtype=np.int64)
    for t in range(a.shape[0] - 1):
        np.add.at(counts, (a[t], a[t + 1]), 1)
    row = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row, out=np.zeros(counts.shape, float), where=row > 0)
    return counts, probs


def motif_lifetimes(probs, dt_ns=5.0):
    """Mean residence time per motif, tau_m = dt / (1 - P_mm).

    dt_ns = 5.0 is the trajectory sampling interval (timeKeeper.csv).

    Two caveats that belong in the text alongside the numbers:
      * tau is bounded below by dt. A motif whose true lifetime is shorter than
        the 5 ns sampling interval cannot be resolved, and comes out as ~5 ns.
      * these are non-equilibrium residence times along a quench, not
        equilibrium lifetimes; they are not interconvertible with rates.
    P_mm = 1 (a motif never observed to leave) returns inf, deliberately.
    """
    p = np.asarray(probs, dtype=float)
    diag = np.diag(p)
    with np.errstate(divide="ignore"):
        tau = dt_ns / (1.0 - diag)
    return tau


# ============================================================ spatial grammar

def cooccurrence_matrix(labels, neighbour_idx, n_motifs=18, one_based=True,
                        k=None, normalise="row"):
    """How often motif i is found next to motif j, over a neighbour table.

    `neighbour_idx` is the (n_atoms, k) table from `rnn_neighbour_table`;
    -1 padding is masked out rather than wrapping to the last atom.
    normalise: "row" (per-motif conditional), "total", or None (raw counts).
    """
    lab = np.asarray(labels, dtype=np.int64)
    if one_based:
        lab = lab - 1
    nb = np.asarray(neighbour_idx)[:, :k] if k else np.asarray(neighbour_idx)
    m = np.zeros((n_motifs, n_motifs), dtype=np.int64)
    for i in range(nb.shape[0]):
        js = nb[i]
        js = js[js >= 0]
        if js.size:
            np.add.at(m[lab[i]], lab[js], 1)
    if normalise == "row":
        row = m.sum(axis=1, keepdims=True)
        return np.divide(m, row, out=np.zeros(m.shape, float), where=row > 0)
    if normalise == "total":
        return m / m.sum() if m.sum() else m.astype(float)
    return m


def neighbour_average(values, neighbour_idx, kmin=0, kmax=4):
    """Mean of `values` over neighbours [kmin, kmax) of each atom.

    -1 padding is masked; an atom with no valid neighbour in the window gets
    NaN rather than a silently wrong number. The original indexed with the raw
    -1 and so averaged in the frame's last atom once per missing slot.
    """
    v = np.asarray(values, dtype=float)
    nb = np.asarray(neighbour_idx)[:, kmin:kmax]
    out = np.full(nb.shape[0], np.nan)
    for i in range(nb.shape[0]):
        js = nb[i]
        js = js[js >= 0]
        if js.size:
            out[i] = v[js].mean()
    return out


def neighbour_class_counts(labels, neighbour_idx, n_classes=None,
                           kmin=0, kmax=40, one_based=True):
    """Count of each class among neighbours [kmin, kmax) of each atom.

    Returns (n_atoms, n_classes). Same -1 masking as `neighbour_average`.
    """
    lab = np.asarray(labels, dtype=np.int64)
    if one_based:
        lab = lab - 1
    n_classes = int(lab.max()) + 1 if n_classes is None else n_classes
    nb = np.asarray(neighbour_idx)[:, kmin:kmax]
    out = np.zeros((nb.shape[0], n_classes), dtype=np.int64)
    for i in range(nb.shape[0]):
        js = nb[i]
        js = js[js >= 0]
        if js.size:
            np.add.at(out[i], lab[js], 1)
    return out


# ============================================================ per-motif stats

def motif_property_table(labels, properties, one_based=True):
    """Per-motif count, fraction, mean, std, median and IQR of each property.

    `properties` maps a column name to a 1-D array over atoms. This is what
    Tables S2-S6 report, and `iqr` is what settles the "slightly broader
    qbar_12" claim with a number instead of an eyeball (todo B8).
    """
    lab = np.asarray(labels, dtype=np.int64)
    if one_based:
        lab = lab - 1
    n = int(lab.max()) + 1
    rows = []
    for m in range(n):
        sel = lab == m
        row = {"motif": m + 1 if one_based else m,
               "count": int(sel.sum()),
               "fraction_pct": 100.0 * sel.sum() / lab.size}
        for name, vals in properties.items():
            v = np.asarray(vals, dtype=float)[sel]
            v = v[np.isfinite(v)]
            if v.size:
                q1, q3 = np.percentile(v, [25, 75])
                row.update({f"{name}_mean": v.mean(), f"{name}_std": v.std(),
                            f"{name}_median": np.median(v), f"{name}_iqr": q3 - q1})
            else:
                row.update({f"{name}_{s}": np.nan
                            for s in ("mean", "std", "median", "iqr")})
        rows.append(row)
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


def voronoi_volume_to_density(v_voro_A3, molar_mass=18.01528):
    """Voronoi cell volume (A^3 per water) -> local density in g/cm^3."""
    n_a = 6.02214076e23
    return molar_mass / (np.asarray(v_voro_A3, dtype=float) * 1e-24 * n_a)
