"""3. Encode to the latent Z, then project to 2-D (PCA then UMAP).

    python scripts/03_project.py

Writes `Z.npy` and `proj/`: the 2-D embedding **and the fitted reducer**. Keep
the reducer. Projecting a second trajectory THROUGH it is a different
experiment from fitting a new one, and having the file on disk is what makes
clear which of the two you did.

The other slow step. UMAP is fitted on a subsample and applied to the rest, and
its result depends on the thread count as well as the seed -- so an exact rerun
means the same `n_jobs`, not just the same `--seed`.
"""
import subprocess, sys
from config import OUT, need_stage

need_stage(f"{OUT}/vae", "scripts/02_train_vae.py")
for stage, args in [
    ("encode", ["--features", f"{OUT}/features", "--run", f"{OUT}/vae",
                "--out", f"{OUT}/Z.npy"]),
    ("project", ["--z", f"{OUT}/Z.npy", "--out", f"{OUT}/proj"])]:
    rc = subprocess.run([sys.executable, "-m", "icemotifs.cli", stage] + args).returncode
    if rc:
        sys.exit(rc)
