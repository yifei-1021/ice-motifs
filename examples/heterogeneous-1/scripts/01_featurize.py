"""1. Featurise: Steinhardt + SOAP per atom, cutoffs scaled by this system's d_nn.

    python scripts/01_featurize.py /path/to/interface.dump (LAMMPS dump)

Writes `features/`: 281 columns per atom -- 27 Steinhardt order parameters, 252
SOAP components, the Voronoi volume and the per-atom potential energy.

**Read the d_nn line it prints before trusting anything downstream.** Every
cutoff here is a fixed multiple of the first-neighbour distance -- the SOAP
radius, all three Steinhardt shells, the ring network -- so the stage measures
d_nn from your own frames and stops if it is more than 3% from the registry
value for this system. Nothing crashes on a wrong d_nn and nothing downstream
can undo one: the motifs simply describe a length scale you did not intend. If
this state point really does differ, pass --d-nn deliberately and say which
value you used.
"""
import subprocess, sys
from config import SYSTEM, KIND, STRIDE, OUT, FEATURIZE, flags, need_input

inp = need_input(sys.argv)
subprocess.run([sys.executable, "-m", "icemotifs.cli", "check",
                "--system", SYSTEM, "--dump", inp], check=False)
sys.exit(subprocess.run([sys.executable, "-m", "icemotifs.cli", "featurize",
    "--system", SYSTEM, "--input", inp, "--kind", KIND,
    "--out", f"{OUT}/features", "--stride", str(STRIDE)]
    + flags(FEATURIZE)).returncode)
