"""
Ring statistics: three definitions, and why the choice matters.

THE PROBLEM
-----------
The published analysis uses `networkx.cycle_basis`. That returns a *cycle
basis* -- |E| - |V| + c cycles that generate the cycle space -- which is not
the set of rings, and the paper filters it to sizes 3-7 and reports the result
as a ring-size distribution.

Measured on a perfect diamond lattice (topologically ice Ic), where the answer
is known analytically: every atom lies in 12 six-membered rings, so 216 atoms
have exactly 432 six-rings and NOTHING shorter.

    method                  cycles   of size 6   mean size
    truth                      432         432        6.0
    nx.cycle_basis             217          37       27.0
    nx.minimum_cycle_basis     217         214        6.1
    sp_rings (this module)     432         432        6.0

`cycle_basis` recovers 8.6% of the six-rings, and 80%+ of what it returns are
long cycles (up to length 192 on a 512-atom cell) that get discarded by the
size filter. It is also basis-dependent: relabelling the same atoms changes the
size-6 count by +-13%, so two people running the same code on the same frame
get different numbers.

WHAT THAT DOES AND DOES NOT INVALIDATE
--------------------------------------
Across an order-to-disorder series, the `cycle_basis` statistics still track
the correct ones with Spearman rho = +1.000 for both the hexamer fraction and
the mean ring size. So COMPARATIVE claims -- trimers dominating the liquid,
hexamers dominating after growth, co-occurrence changing over the trajectory --
survive the choice. ABSOLUTE numbers do not, and neither does any comparison
with the ring-statistics literature, which means SP/primitive rings.

IT IS NOT A BUG, AND IT IS NOT PERIODIC BOUNDARIES
--------------------------------------------------
Both were tested. With open boundaries the picture is the same (29 of 118 basis
cycles are six-rings, longest 58). And `nx.cycle_basis` passes every validity
check: the cycles are genuine closed walks, the count equals |E|-|V|+c, and they
are linearly independent over GF(2) (rank 217 of 217). It does exactly what
Paton's algorithm promises -- a spanning tree plus one non-tree edge per cycle,
where the tree path can be long.

The reason no basis works is dimensional. The cycle space here has dimension
|E|-|V|+c = 217, while the structure contains 432 six-rings whose GF(2) rank is
214: they span the space but are 218-fold linearly dependent. Any basis holds at
most ~217 of them. Ring counting is a combinatorial enumeration, not a minimal
generating set, so it cannot be obtained from a basis at all -- which is why
`minimum_cycle_basis`, at 214 of a possible 217, is as good as a basis can get
and still finds only half the rings.

PBC does matter in one way: a cycle can close by wrapping the box, which needs
at least L/d_nn edges. For a 160 A box that is ~58 edges, well above a size-7
cut. For cells below ~17 A a 7-ring could wrap, and this search would need a
minimum-image guard.

WHICH TO USE
------------
`sp_rings` is the shortest-path (King) definition, and it is both correct and
cheap: bounded at size 7 the search never leaves a small neighbourhood, so it
is linear in system size -- about 39 s per frame at 128,000 atoms, roughly 90
minutes for a 141-frame trajectory on one core, and trivially parallel. `minimum_cycle_basis` is also
correct but is O(N^3): 18 s at 216 atoms, which does not reach 128,000.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np


def sp_rings(g, max_size=7):
    """Shortest-path (King) rings up to `max_size`, as a set of frozensets.

    For each atom u and each pair of its neighbours (v, w), every shortest path
    from v to w that avoids u closes a ring through u. Keeping only paths whose
    length equals the true shortest distance is King's criterion.

    All shortest paths are enumerated, not just the first found, so rings that
    are degenerate in length are counted once each rather than missed.
    """
    rings = set()
    for u in g:
        nbrs = sorted(g[u])
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v, w = nbrs[i], nbrs[j]
                for path in _bounded_shortest_paths(g, v, w, u, max_size - 2):
                    cyc = frozenset(path) | {u}
                    if len(cyc) == len(path) + 1:
                        rings.add(cyc)
    return rings


def _bounded_shortest_paths(g, src, dst, avoid, max_edges):
    """All shortest src->dst paths avoiding `avoid`, at most max_edges long."""
    dist = {src: 0}
    preds = {src: []}
    frontier = deque([src])
    found_at = None
    while frontier:
        x = frontier.popleft()
        if found_at is not None and dist[x] >= found_at:
            continue
        if dist[x] >= max_edges:
            continue
        for y in g[x]:
            if y == avoid:
                continue
            if y not in dist:
                dist[y] = dist[x] + 1
                preds[y] = [x]
                if y == dst:
                    found_at = dist[y]
                else:
                    frontier.append(y)
            elif dist[y] == dist[x] + 1:
                preds[y].append(x)
    if dst not in dist or dist[dst] > max_edges:
        return

    def walk(node):
        if node == src:
            yield [src]
            return
        for p in preds[node]:
            for tail in walk(p):
                yield tail + [node]

    yield from walk(dst)


def cycle_basis_rings(g, max_size=7):
    """What the published analysis computes. Kept for reproduction only."""
    import networkx as nx
    return [c for c in nx.cycle_basis(g) if len(c) <= max_size]


def minimum_cycle_basis_rings(g, max_size=7):
    """Deterministic shortest basis. Correct here, but O(N^3) -- small cells only."""
    import networkx as nx
    return [c for c in nx.minimum_cycle_basis(g) if len(c) <= max_size]


def distribution(rings, sizes=(3, 4, 5, 6, 7)):
    c = Counter(len(r) for r in rings)
    out = {s: c.get(s, 0) for s in sizes}
    total = sum(out.values())
    out["total"] = total
    out["mean_size"] = (sum(s * n for s, n in out.items() if isinstance(s, int))
                        / total) if total else float("nan")
    return out


def compare(g, max_size=7):
    """Run all three on one graph. This is the sensitivity check for SI S2.6."""
    return {
        "sp_rings": distribution(sp_rings(g, max_size)),
        "cycle_basis": distribution(cycle_basis_rings(g, max_size)),
        "minimum_cycle_basis": distribution(minimum_cycle_basis_rings(g, max_size)),
    }


# --------------------------------------------------------------------- sanity

def neighbour_graph(frame, r_cut):
    """Radius-nearest-neighbour graph of an ASE frame, honouring PBC."""
    import networkx as nx
    from ase.neighborlist import neighbor_list

    i, j = neighbor_list("ij", frame, float(r_cut))
    g = nx.Graph()
    g.add_nodes_from(range(len(frame)))
    g.add_edges_from(zip(i.tolist(), j.tolist()))
    return g


def tetrahedral_sanity(g, label="frame", verbose=True):
    """Flag rings that a tetrahedral network cannot physically contain.

    WHY THIS EXISTS. In a 4-coordinated tetrahedral network the neighbour-to-
    neighbour distance is ~1.63 d_nn (4.5 A for water, 3.8 A for Si), so at a
    cutoff that captures only the first shell there are NO triangles and no
    4-cycles at all. Ideal ice Ih, ice Ic and diamond contain six-rings and
    nothing shorter.

    So a non-zero 3- or 4-cycle count on a structure you believe is crystalline
    is not a ring result: it means the neighbour cutoff has reached into the
    second shell. This was found in the released polymorph ring table, where
    POSCAR.Ih.opt is clean (23,835 six-rings, nothing else) but POSCAR.Ic.opt
    reports 1,109 trimers and 256 tetramers -- from the same code at the same
    cutoff, on a denser relaxed cell.

    Returns (n_triangles, n_squares). Cheap: triangles come straight from the
    adjacency, no ring enumeration needed.
    """
    import networkx as nx

    tri = sum(nx.triangles(g).values()) // 3
    squares = 0
    for c in nx.cycle_basis(g):          # cheap upper bound on 4-cycles
        if len(c) == 4:
            squares += 1

    if verbose and (tri or squares):
        deg = np.mean([d for _, d in g.degree()]) if len(g) else 0.0
        print(f"  !! {label}: {tri} triangle(s) and >= {squares} 4-cycle(s) "
              f"at mean degree {deg:.2f}.")
        print(f"     A tetrahedral network at a first-shell cutoff has neither. "
              f"Lower the ring cutoff (try the first MINIMUM of g(r), not a "
              f"fixed multiple of the first peak) and re-check before trusting "
              f"any ring count from this frame.")
    return tri, squares


def cutoff_scan(frame, cutoffs, max_size=7):
    """Ring-size distribution as a function of the neighbour cutoff.

    The one-command version of "is 3.2 A right for this cell?". Run it on a
    structure whose answer you know (Ih, Ic, diamond): the correct cutoff is the
    largest one that still gives zero triangles.
    """
    rows = []
    for rc in cutoffs:
        g = neighbour_graph(frame, rc)
        tri, sq = tetrahedral_sanity(g, label=f"r_cut={rc:.2f}", verbose=False)
        d = distribution(sp_rings(g, max_size=max_size))
        rows.append({"r_cut": float(rc),
                     "mean_degree": float(np.mean([x for _, x in g.degree()])),
                     "triangles": int(tri), "four_cycles": int(sq),
                     **{f"R{k}": int(v) for k, v in d.items()}})
    return rows
