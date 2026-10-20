"""1. Featurise: Steinhardt + SOAP per atom, cutoffs scaled by this system's d_nn.

    python scripts/01_featurize.py /path/to/llt.extxyz (multi-frame extxyz)

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

One line of the output is expected noise. The script runs `icemotifs check`
first for its d_nn read-out, and that stage reads LAMMPS dumps only -- so on
this folder's input it prints `frames in file: 0` and gives up. Nothing is
wrong: stage 1 does its own d_nn handling either way, and the numbers it prints
below that are the ones that count.
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
