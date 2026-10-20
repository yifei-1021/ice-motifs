"""5. Rank the motifs along the projection axis that carries the physics.

    python scripts/05_relabel.py

Writes `relabel.csv` and the colour legend. **AXIS = 1 for this dataset.**

The axis is not a default and not a convention: it is whichever axis the
structure varies along here, and it differs between datasets. Ranking on the
wrong one produces a figure that looks entirely reasonable and orders the motifs
by nothing. For a dataset of your own, choose it with evidence -- pass
`--axis-from a_property.npy` to `icemotifs relabel` and read the correlation it
prints for both axes.

`motif` identifies; `motif_relabel` orders. Both are kept, because a colour ramp
needs something ordered to follow and the vocabulary number is not it.
"""
import subprocess, sys
from config import OUT, AXIS, need_stage

need_stage(f"{OUT}/motifs", "scripts/04_cluster.py")
sys.exit(subprocess.run([sys.executable, "-m", "icemotifs.cli", "relabel",
    "--embedding", f"{OUT}/proj/embedding.npy",
    "--labels", f"{OUT}/motifs/labels.npy",
    "--out", f"{OUT}/motifs", "--axis", str(AXIS)]).returncode)
