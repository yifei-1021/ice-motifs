#!/usr/bin/env python3
"""
STAGE 0 -- measure the first-neighbour distance before anything is featurised.

    icemotifs check --system silicon-sw --dump si_quench.lammpstrj

Every cutoff in the pipeline is a fixed multiple of `d_nn`: the SOAP radius,
all three Steinhardt shells, the ring network and the coordination cutoff. A
wrong `d_nn` rescales them together, nothing crashes, and the motifs describe a
length scale you did not intend -- so this stage prints the registry's value
beside the one measured from frame 0 of your own trajectory, and says so when
they disagree by more than 5%.

It reads one frame only. The reader itself lives in `icemotifs/lammps.py`;
this is the stage that reports what it found.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from icemotifs import lammps as L           # noqa: E402
from icemotifs import systems as S          # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="icemotifs check",
        description="measure d_nn from a LAMMPS dump, before featurising")
    p.add_argument("--dump", required=True, help="LAMMPS dump file")
    p.add_argument("--system", required=True,
                   help="a registered system (`icemotifs systems` lists them)")
    a = p.parse_args(argv)

    sysdef = S.get(a.system)
    print(sysdef.describe())
    print(f"\nframes in file: {L.count_frames(a.dump)}")
    for atoms, energy in L.iter_dump(a.dump, sysdef, allow_missing_energy=True,
                                     max_frames=1):
        print(f"frame 0: {len(atoms)} atoms")
        print(f"  cell diag      {np.diag(atoms.get_cell())}")
        print(f"  energy range   {energy.min():.4f} .. {energy.max():.4f}")
        d = S.measure_d_nn(atoms)
        print(f"  measured d_nn  {d:.3f} A   (config says {sysdef.d_nn})")
        if abs(d - sysdef.d_nn) / sysdef.d_nn > 0.05:
            print("  !! differs from the configured value by more than 5% --"
                  " use the measured one and say so in the SI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
