"""Stage 7: write motif-labelled frames for a renderer.

    icemotifs frames --system S --input TRAJ --kind {dump,extxyz,filelist} \
                     --labels runs/X/motifs/labels.npy \
                     --relabel runs/X/motifs/relabel.csv \
                     --out runs/X/frames --stride 20 [--every 1]

Writes extxyz carrying two per-atom columns beside the positions:

    motif           the vocabulary number -- the identifier everything else uses
    motif_relabel   its rank along the chosen projection axis -- what a
                    continuous colour ramp can follow

Both are written because neither does the other's job: see icemotifs.relabel.

THE ONE THING THAT SILENTLY GOES WRONG
--------------------------------------
Atom order. Row j of `labels.npy` must be the same atom as row j of the frame
read back here, or every label lands on the wrong atom and the picture looks
perfectly plausible while being meaningless. Nothing downstream can detect it.

So this stage REFUSES to write unless the arithmetic is exact: the label count
must equal (frames written) x (atoms per frame), using the same `--system`,
`--stride` and input that produced the labels. If you strided differently when
featurising, the counts will not match and you will get an error rather than a
wrong picture. That check is the whole reason this is a stage and not a
three-line script.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def _read_relabel(path):
    if not path:
        return None
    with open(path) as fh:
        return {int(r["motif"]): int(r["motif_relabel"]) for r in csv.DictReader(fh)}


def _frames(input_path, kind):
    """Yield ase.Atoms for each frame of the trajectory, in file order."""
    from ase.io import iread, read
    if kind == "filelist":
        with open(input_path) as fh:
            paths = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        for p in paths:
            yield read(p)
    elif kind == "dump":
        for at in iread(input_path, format="lammps-dump-text", index=":"):
            yield at
    else:
        for at in iread(input_path, index=":"):
            yield at


def main(argv=None):
    p = argparse.ArgumentParser(prog="icemotifs frames")
    p.add_argument("--input", required=True)
    p.add_argument("--kind", default="dump", choices=("dump", "extxyz", "filelist"))
    p.add_argument("--labels", required=True, help="motifs/labels.npy")
    p.add_argument("--relabel", help="motifs/relabel.csv (adds motif_relabel)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=20,
                   help="MUST match the stride used when featurising")
    p.add_argument("--every", type=int, default=1,
                   help="write only every Nth of the strided frames")
    p.add_argument("--one-file", action="store_true",
                   help="also concatenate into frames.all.extxyz")
    a = p.parse_args(argv)

    labels = np.load(a.labels)
    rank = _read_relabel(a.relabel)

    kept = [at for i, at in enumerate(_frames(a.input, a.kind)) if i % a.stride == 0]
    if not kept:
        print("no frames read; check --input and --kind", file=sys.stderr)
        return 2
    n_at = len(kept[0])
    if any(len(at) != n_at for at in kept):
        print("frames differ in atom count; this stage assumes a fixed-N "
              "trajectory.", file=sys.stderr)
        return 2

    expect = len(kept) * n_at
    if len(labels) != expect:
        print(f"REFUSING TO WRITE: {len(labels):,} labels but {len(kept)} frames "
              f"x {n_at} atoms = {expect:,}.", file=sys.stderr)
        print("The labels were produced from a different stride, input or "
              "system. Re-run featurise and cluster with the same --stride, or "
              "pass the stride that was used.", file=sys.stderr)
        return 2

    os.makedirs(a.out, exist_ok=True)
    from ase.io import write
    written = []
    for fi, at in enumerate(kept):
        if fi % a.every:
            continue
        lab = labels[fi * n_at:(fi + 1) * n_at]
        at.set_array("motif", np.asarray(lab, dtype=np.int32))
        if rank is not None:
            at.set_array("motif_relabel",
                         np.array([rank.get(int(v), 0) for v in lab], dtype=np.int32))
        path = os.path.join(a.out, f"frame_{fi * a.stride:06d}.extxyz")
        write(path, at, format="extxyz")
        written.append(path)

    print(f"  {len(written)} frames x {n_at} atoms -> {a.out}")
    print(f"  columns: motif{', motif_relabel' if rank is not None else ''}")
    if rank is None:
        print("  (no --relabel given, so no motif_relabel column; a continuous "
              "colour ramp has nothing ordered to follow without it)")

    if a.one_file and written:
        allp = os.path.join(a.out, "frames.all.extxyz")
        with open(allp, "w") as out:
            for path in written:
                with open(path) as fh:
                    out.write(fh.read())
        print(f"  concatenated -> {allp}")
        print("  NOTE: this joins exactly the frames just written. If you change "
              "the frame count later, clear the directory first -- a stale "
              "frame_*.extxyz from a previous run would be picked up by any "
              "tool that globs this directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
