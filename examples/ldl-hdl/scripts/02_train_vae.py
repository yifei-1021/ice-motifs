"""2. Train the VAE on the descriptors from step 1.

    python scripts/02_train_vae.py

Writes `vae/`: the weights, the per-block scalers fitted on training rows only,
and the loss history. One of the two slow steps -- put `epochs` into TRAIN_KW
for a smoke run before committing to a real one.

Trained on this trajectory alone; no weights are carried in from anywhere, so
the vocabulary that comes out belongs to this dataset.

**The setting to understand is lambda_pred** (default 0.1, the published
value). It weights a head that predicts per-atom Voronoi volume and energy from
the latent. Neither is an encoder input, but a non-zero weight shapes the latent
so that both can be recovered from it -- so motifs separating on density is
partly by construction. Set it to 0 for a latent shaped by structure alone, and
you must set it to 0 if your trajectory carries no per-atom energy.
"""
import subprocess, sys
from config import SYSTEM, OUT, TRAIN, flags, need_stage

need_stage(f"{OUT}/features", "scripts/01_featurize.py")
sys.exit(subprocess.run([sys.executable, "-m", "icemotifs.cli", "train",
    "--features", f"{OUT}/features", "--system", SYSTEM,
    "--out", f"{OUT}/vae"] + flags(TRAIN)).returncode)
