#!/usr/bin/env python3
"""
STAGE 5 (Supplementary Section 9) -- two-stage clustering: HDBSCAN then k-means.

    python hdbscan_kmeans.py --embedding runs/silicon-sw/proj/embedding.npy \
        --out runs/silicon-sw/motifs

This reproduces the published procedure exactly, and the procedure is the
result, so it is worth stating plainly:

  1. HDBSCAN is run on a 1-in-100 subsample of the 2-D embedding. It resolves
     the well-separated DENSITY MAXIMA -- in the published water run, eight of
     them: five ice motifs and three liquid motifs.

  2. Everything HDBSCAN returns as an OUTLIER is the intermediate region. It is
     not noise: it is a continuum with no density maxima to find, which is why
     a density-based method declines to cluster it. k-means partitions it into
     `--n-kmeans` cells (ten in the published run), giving the transient motifs.

  3. Every point outside the subsample takes the label of its nearest
     neighbour in the subsample.

WHAT THIS MEANS FOR THE NUMBERS YOU REPORT
------------------------------------------
The total motif count is 8 + n_kmeans, and the second term is a RESOLUTION
CHOICE, not a measurement. Do not report "the same number of motifs was
recovered" as evidence of generality -- it cannot be, by construction. What is
meaningful, and what the sensitivity sweep in S2.4 should show, is:

  * how many density maxima stage 1 resolves          <- a property of the data
  * the fraction of points returned as outliers       <- the size of the continuum
  * whether the liquid / transient / ordered partition survives

all three of which this script reports.

The subsample exists to keep HDBSCAN tractable, not to change the answer;
`--subsample 1` runs on everything if you can afford it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

# NOTE: the original carried a three-directories-up sys.path.insert so
# it could find the library when run from inside the source tree. It is
# removed here: the package is installed, and leaving it in would let
# whatever sits three directories above shadow the installed icemotifs.

PUBLISHED = dict(min_cluster_size=1000, min_samples=1, subsample=100,
                 n_kmeans=10, seed=123456)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subsample", type=int, default=PUBLISHED["subsample"],
                   help="take every Nth point for the clustering (published: 100)")
    p.add_argument("--min-cluster-size", type=int,
                   default=PUBLISHED["min_cluster_size"])
    p.add_argument("--min-samples", type=int, default=PUBLISHED["min_samples"])
    p.add_argument("--n-kmeans", type=int, default=PUBLISHED["n_kmeans"],
                   help="cells for the intermediate continuum (published: 10)")
    p.add_argument("--seed", type=int, default=PUBLISHED["seed"])
    a = p.parse_args(argv)

    from icemotifs.cluster import two_stage, assign_rest

    E = np.load(a.embedding)
    print(f"embedding {E.shape} from {a.embedding}")
    os.makedirs(a.out, exist_ok=True)

    sub = E[::a.subsample]
    print(f"clustering on every {a.subsample}th point: {len(sub):,} of {len(E):,}")

    part = two_stage(sub, min_cluster_size=a.min_cluster_size,
                     min_samples=a.min_samples, n_kmeans=a.n_kmeans,
                     seed=a.seed)
    print(f"\nstage 1 + 2 ({part.implementation})")
    print(f"  density maxima resolved : {part.n_density_maxima}")
    print(f"  outliers (intermediate) : {100*part.outlier_fraction:.1f}% "
          f"-> {part.n_kmeans} k-means cells")
    for c, n in sorted(part.counts.items()):
        kind = "density max" if c < part.n_density_maxima else "k-means cell"
        if c == -1:
            kind = "unassigned"
        print(f"    motif {c:>3} ({kind:<12}): {n:>8,} points")

    labels, knn = assign_rest(E, sub, part.labels)
    print(f"\nstage 3: nearest-point assignment for all {len(E):,} points")

    np.save(os.path.join(a.out, "labels.npy"), labels)
    with open(os.path.join(a.out, "clusterer.pkl"), "wb") as fh:
        pickle.dump({"knn": knn, "sub_labels": part.labels,
                     "subsample": a.subsample}, fh)

    counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    meta = dict(
        embedding=os.path.abspath(a.embedding), n_rows=int(len(E)),
        implementation=part.implementation, subsample=a.subsample,
        hdbscan=dict(min_cluster_size=a.min_cluster_size,
                     min_samples=a.min_samples,
                     max_cluster_size=len(sub) // 2),
        n_density_maxima=part.n_density_maxima,
        outlier_fraction=round(part.outlier_fraction, 4),
        n_kmeans=part.n_kmeans, n_motifs=part.n_motifs,
        seed=a.seed, counts=counts,
    )
    with open(os.path.join(a.out, "clustering.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n{part.summary()}")
    print(f"labels {labels.shape} -> {a.out}/labels.npy")
    print("\nReport for the SI: the number of DENSITY MAXIMA "
          f"({part.n_density_maxima}) and the OUTLIER FRACTION "
          f"({100*part.outlier_fraction:.1f}%), not the motif total -- the "
          "total depends on --n-kmeans, which is a choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
