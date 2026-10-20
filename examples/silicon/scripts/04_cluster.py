"""4. Vector-quantise the projection: HDBSCAN clusters, then k-means tiles.

    python scripts/04_cluster.py

Writes `motifs/`: `labels.npy` and `clustering.json`. **Quote two numbers from
that json -- the HDBSCAN cluster count and the intermediate fraction.** Its
field names are still `n_density_maxima` and `outlier_fraction`, which is what
the lines below print; they are the same two numbers.

The two stages produce two KINDS of label and they do not mean the same thing.
An HDBSCAN cluster is a region the data actually concentrates in. A k-means cell
is a tile over the intermediate regions -- the continuum between clusters, which
has no density peak for HDBSCAN to find. So the motif total is a measured number
plus your own choice of k, and reports a parameter as much as a result.

`min_cluster_size` and `n_kmeans` in CLUSTER_KW are the two knobs. On the
ten-frame sample the published values resolve far less than the published run
did -- that is the sample size, not a failure.
"""
import json, subprocess, sys
from config import OUT, need_stage

need_stage(f"{OUT}/proj", "scripts/03_project.py")
rc = subprocess.run([sys.executable, "-m", "icemotifs.cli", "cluster",
    "--embedding", f"{OUT}/proj/embedding.npy", "--out", f"{OUT}/motifs"]).returncode
if rc:
    sys.exit(rc)
meta = json.load(open(f"{OUT}/motifs/clustering.json"))
print(f"\n  density maxima    {meta.get('n_density_maxima')}")
print(f"  outlier fraction  {meta.get('outlier_fraction')}")
print(f"  k-means cells     {meta.get('n_kmeans')}")
print("  Quote the first two. The motif total is maxima + cells.")
