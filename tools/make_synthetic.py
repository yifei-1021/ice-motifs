#!/usr/bin/env python3
"""Generate a tiny synthetic LAMMPS dump so the pipeline can be exercised
without the real trajectories. Diamond-like lattice + noise, two 'phases'."""
import numpy as np, sys, os

def dump(path, n_frames=6, nx=6, a=5.43, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],
                      [.25,.25,.25],[.25,.75,.75],[.75,.25,.75],[.75,.75,.25]])
    cells = np.array([[i,j,k] for i in range(nx) for j in range(nx) for k in range(nx)])
    pos0 = ((cells[:,None,:]+basis[None,:,:]).reshape(-1,3))*a
    L = nx*a; n = len(pos0)
    # Atom ids are assigned ONCE and never change -- that is what a real LAMMPS
    # dump does. Re-drawing them per frame (an earlier version of this file did)
    # makes row i a different particle in every frame, which is exactly the bug
    # the reader's id-sorting exists to prevent, so the test data must not have it.
    ids = rng.permutation(n)+1
    with open(path,'w') as f:
        for t in range(n_frames):
            amp = 0.05 + 0.45*t/(n_frames-1)          # progressively disordered
            pos = (pos0 + rng.normal(0, amp, pos0.shape)) % L
            pe  = -4.6 + 0.35*amp*rng.standard_normal(n)
            f.write("ITEM: TIMESTEP\n%d\n"%(t*1000))
            f.write("ITEM: NUMBER OF ATOMS\n%d\n"%n)
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for _ in range(3): f.write("0.0 %.6f\n"%L)
            f.write("ITEM: ATOMS id type x y z c_pe\n")
            # written in id-scrambled order on purpose: the reader must sort
            order = rng.permutation(n)
            for r in order:
                f.write("%d 1 %.5f %.5f %.5f %.5f\n"%(ids[r],*pos[r],pe[r]))
    return n, n_frames

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv)>1 else "synthetic.lammpstrj"
    n,nf = dump(out)
    print(f"wrote {out}: {nf} frames x {n} atoms")
