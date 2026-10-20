#!/usr/bin/env python3
"""
STAGE 4 (Supplementary Section 9) -- PCA + UMAP projection of the latent Z.

    python pca_umap.py --z runs/silicon-sw/Z.npy --out runs/silicon-sw/proj

Writes:
    embedding.npy        (n_rows, 2)   the 2-D projection every later stage uses
    reducer.pkl                        the fitted UMAP, so new points can be added
    pca.pkl                            the fitted PCA, if --n-pca > 0
    projection.json                    parameters and the fit/transform split

THE REDUCER IS SAVED. THAT IS THE POINT OF THIS SCRIPT.
-------------------------------------------------------
In the published run it was not, because the pickle came to 10+ GB and the fit
was slow. Both problems have the same cause: a UMAP pickle stores the graph of
every point it was FITTED on, so its size scales with the fit set, not with the
data you later project.

The fix is `--fit-subsample`: fit on a few hundred thousand points, then
`transform` the rest in batches. A 200,000-point fit pickles to roughly 200 MB
and takes minutes rather than hours, and because `transform` is deterministic
given the fitted reducer, everything downstream becomes reproducible -- which
is what Supplementary Section 9 needs, since it compares projections across
systems and you cannot compare projections you cannot regenerate.

Subsampling changes the embedding slightly. It is not a shortcut you hide: say
in the SI that the reducer was fitted on N points and applied to the rest.

WHY PCA FIRST IS OPTIONAL
-------------------------
`--n-pca 0` (the default) runs UMAP directly on Z, which is what the published
analysis did. `--n-pca K` inserts a linear PCA to K components first. R2 asks
where PCA sits in the pipeline; the honest answer is that it does not sit in
the motif path at all -- the nonlinear encoders replace it -- and this flag
exists so that claim can be tested rather than asserted. Compare the two with
`S2-descriptor-tests/02_projection_comparison.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np


def _clamp_jobs(n):
    """umap-learn hands n_jobs to numba.set_num_threads, which RAISES if the
    value exceeds the cores actually available -- "The number of threads must
    be between 1 and 2". Under PBS the visible core count is the cpuset, not
    the machine, so asking for 10 on a node that granted 8 is a crash, not a
    slowdown. Clamp, and say so."""
    import os
    try:
        avail = len(os.sched_getaffinity(0))
    except Exception:
        avail = os.cpu_count() or 1
    if n < 0:
        n = avail
    if n > avail:
        print(f"  n_jobs {n} exceeds the {avail} core(s) available here; "
              f"using {avail}")
        n = avail
    return max(1, n)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", required=True, help="latent matrix .npy from stage 3")
    p.add_argument("--out", required=True)
    p.add_argument("--n-pca", type=int, default=0,
                   help="0 = UMAP directly on Z (published behaviour)")
    p.add_argument("--fit-subsample", type=int, default=200000,
                   help="points to FIT the reducer on; 0 = fit on everything "
                        "(this is what produced the 10+ GB pickle)")
    p.add_argument("--transform-batch", type=int, default=500000)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="threads for UMAP. NOTE: umap-learn ignores this "
                        "whenever random_state is set, so n_jobs != 1 "
                        "drops the seed and the embedding stops being "
                        "bit-reproducible. 1 keeps the seed.")
    a = p.parse_args(argv)
    a.n_jobs = _clamp_jobs(a.n_jobs)

    Z = np.load(a.z)
    print(f"Z {Z.shape} from {a.z}")
    os.makedirs(a.out, exist_ok=True)

    X = Z
    pca = None
    if a.n_pca > 0:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=a.n_pca, random_state=a.seed).fit(Z)
        X = pca.transform(Z)
        ev = float(pca.explained_variance_ratio_.sum())
        print(f"PCA -> {X.shape}, {100*ev:.2f}% of variance in {a.n_pca} PCs")
        with open(os.path.join(a.out, "pca.pkl"), "wb") as fh:
            pickle.dump(pca, fh)

    import umap

    n = len(X)
    if a.fit_subsample and a.fit_subsample < n:
        rng = np.random.default_rng(a.seed)
        fit_idx = np.sort(rng.choice(n, a.fit_subsample, replace=False))
        print(f"fitting the reducer on {len(fit_idx):,} of {n:,} points "
              f"({100*len(fit_idx)/n:.1f}%)")
    else:
        fit_idx = np.arange(n)
        print(f"fitting the reducer on all {n:,} points -- expect a large "
              f"pickle; see --fit-subsample")

    t0 = time.time()
    # umap-learn IGNORES n_jobs whenever random_state is set: it forces a single
    # thread and warns "n_jobs value ... overridden to 1 by setting
    # random_state". So parallel and bit-reproducible are mutually exclusive
    # here, and asking for both silently gives you the slow one. Choose
    # explicitly: --n-jobs 1 keeps the seed and the exact embedding; anything
    # else drops the seed and says so. The fitted reducer is pickled either
    # way, so the embedding stays reapplicable to new points even when the fit
    # itself is not bit-reproducible.
    umap_seed = a.seed
    if a.n_jobs != 1:
        umap_seed = None
        print(f"  UMAP: n_jobs={a.n_jobs}, so random_state is dropped. The "
              f"embedding will NOT be bit-reproducible.\n"
              f"        Use --n-jobs 1 if you need the exact published "
              f"embedding; the saved reducer.pkl still reproduces the mapping "
              f"of new points either way.")
    reducer = umap.UMAP(n_neighbors=a.n_neighbors, min_dist=a.min_dist,
                        n_components=a.n_components, metric=a.metric,
                        random_state=umap_seed, n_jobs=a.n_jobs,
                        verbose=True)
    reducer.fit(X[fit_idx])
    t_fit = time.time() - t0
    print(f"fit: {t_fit:.1f}s")

    rpath = os.path.join(a.out, "reducer.pkl")
    with open(rpath, "wb") as fh:
        pickle.dump(reducer, fh)
    size_mb = os.path.getsize(rpath) / 1e6
    print(f"reducer -> {rpath}  ({size_mb:.0f} MB)")
    if size_mb > 2000:
        print("  !! over 2 GB. Lower --fit-subsample; this is the problem that "
              "stopped the reducer being saved for the published run.")

    t0 = time.time()
    emb = np.empty((n, a.n_components), dtype=np.float32)
    emb[fit_idx] = reducer.embedding_.astype(np.float32)
    rest = np.setdiff1d(np.arange(n), fit_idx, assume_unique=True)
    for i in range(0, len(rest), a.transform_batch):
        chunk = rest[i:i + a.transform_batch]
        emb[chunk] = reducer.transform(X[chunk]).astype(np.float32)
        print(f"  transformed {min(i+a.transform_batch, len(rest)):,}/"
              f"{len(rest):,}", flush=True)
    t_tr = time.time() - t0

    np.save(os.path.join(a.out, "embedding.npy"), emb)
    meta = dict(z=os.path.abspath(a.z), n_rows=int(n), n_pca=a.n_pca,
                fit_subsample=int(len(fit_idx)),
                # Record the EFFECTIVE random_state, not the requested one.
                # umap-learn drops random_state whenever n_jobs != 1, so writing
                # a.seed here claimed `random_state: 123456` for fits that were
                # never seeded at all -- and anyone reading projection.json would
                # conclude the embedding was reproducible when it is not. n_jobs
                # is recorded alongside it because it is what decides this.
                umap=dict(n_neighbors=a.n_neighbors, min_dist=a.min_dist,
                          n_components=a.n_components, metric=a.metric,
                          random_state=umap_seed,
                          random_state_requested=a.seed,
                          n_jobs=a.n_jobs,
                          deterministic=bool(umap_seed is not None)),
                reducer_mb=round(size_mb, 1),
                seconds=dict(fit=round(t_fit, 1), transform=round(t_tr, 1)))
    with open(os.path.join(a.out, "projection.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nembedding {emb.shape} -> {a.out}/embedding.npy")
    print(f"transform: {t_tr:.1f}s")
    print("\nnext: python ../5-clustering/hdbscan_kmeans.py "
          f"--embedding {a.out}/embedding.npy --out {a.out}/../motifs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
