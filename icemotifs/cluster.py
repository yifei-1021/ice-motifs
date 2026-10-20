"""
The published two-stage clustering, in one place.

Stage 1  HDBSCAN on the (subsampled) embedding. It resolves the well-separated
         DENSITY MAXIMA and returns everything else as outliers.
Stage 2  k-means over those outliers. They are not noise -- they are the
         intermediate continuum, which has no density maxima for a
         density-based method to find, so it is partitioned into cells.
Stage 3  nearest-point assignment for every point outside the subsample.

Every script that clusters imports this, so the procedure cannot drift between
the published reproduction, the sensitivity sweeps and the new systems.

The quantity to report is the number of DENSITY MAXIMA and the OUTLIER
FRACTION. The motif total is `n_maxima + n_kmeans`, and the second term is a
resolution choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PUBLISHED = dict(min_cluster_size=1000, min_samples=1, n_kmeans=10,
                 subsample=100, seed=123456)


@dataclass
class Partition:
    labels: np.ndarray               # one label per input row
    n_density_maxima: int
    outlier_fraction: float
    n_kmeans: int
    implementation: str
    counts: dict = field(default_factory=dict)

    @property
    def n_motifs(self) -> int:
        return self.n_density_maxima + self.n_kmeans

    def summary(self) -> str:
        return (f"{self.n_density_maxima} density maxima + {self.n_kmeans} "
                f"k-means cells = {self.n_motifs} motifs; "
                f"{100*self.outlier_fraction:.1f}% of points were outliers "
                f"at stage 1 ({self.implementation})")


def _hdbscan(E, min_cluster_size, min_samples, max_cluster_size):
    try:
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=min_cluster_size,
                     min_samples=min_samples,
                     max_cluster_size=max_cluster_size,
                     store_centers="centroid")
        return cl.fit_predict(E), "sklearn.cluster.HDBSCAN"
    except ImportError:
        import hdbscan as H
        cl = H.HDBSCAN(min_cluster_size=min_cluster_size,
                       min_samples=min_samples)
        return cl.fit_predict(E), "hdbscan (contrib)"


def two_stage(E, *, min_cluster_size=PUBLISHED["min_cluster_size"],
              min_samples=PUBLISHED["min_samples"],
              n_kmeans=PUBLISHED["n_kmeans"],
              seed=PUBLISHED["seed"],
              max_cluster_size=None) -> Partition:
    """Cluster a 2-D (or n-D) embedding. `E` is already subsampled."""
    from sklearn.cluster import KMeans

    E = np.asarray(E, dtype=np.float64)
    if max_cluster_size is None:
        max_cluster_size = len(E) // 2

    lab, impl = _hdbscan(E, min_cluster_size, min_samples, max_cluster_size)
    maxima = sorted(set(int(x) for x in lab) - {-1})
    out = lab == -1
    frac = float(out.mean())

    full = np.asarray(lab, dtype=np.int64).copy()
    # relabel the maxima to 0..k-1 so the k-means cells append cleanly
    remap = {c: i for i, c in enumerate(maxima)}
    for c, i in remap.items():
        full[lab == c] = i

    used_k = 0
    if out.sum() >= n_kmeans and n_kmeans > 0:
        km = KMeans(n_clusters=n_kmeans, random_state=seed, n_init=10)
        full[out] = len(maxima) + km.fit_predict(E[out])
        used_k = n_kmeans
    elif out.sum():
        # Not an error: a system whose intermediate region is empty is a
        # result. Leave those points as -1 and say so.
        full[out] = -1

    counts = {int(c): int((full == c).sum()) for c in np.unique(full)}
    return Partition(labels=full, n_density_maxima=len(maxima),
                     outlier_fraction=frac, n_kmeans=used_k,
                     implementation=impl, counts=counts)


def assign_rest(E_full, E_sub, sub_labels, *, batch=1_000_000):
    """Stage 3: give every point the label of its nearest subsampled point."""
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=1).fit(E_sub, sub_labels)
    out = np.empty(len(E_full), dtype=np.int32)
    for i in range(0, len(E_full), batch):
        out[i:i + batch] = knn.predict(E_full[i:i + batch])
    return out, knn
