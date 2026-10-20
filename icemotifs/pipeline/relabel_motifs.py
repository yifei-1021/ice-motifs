"""Stage 6: rank the motifs along one projection axis and write a colour legend.

    icemotifs relabel --embedding runs/X/proj/embedding.npy \
                      --labels    runs/X/motifs/labels.npy \
                      --out       runs/X/motifs --axis 0

Writes `relabel.csv` (motif, motif_relabel, median_axis, n_atoms,
percent_of_atoms) and `relabel.json` (the parameters, so a figure can be traced
back to the axis it was coloured by).

This stage adds NO new science. It assigns each motif a rank along one axis so a
continuous colour ramp has something ordered to follow -- see icemotifs.relabel
for why the vocabulary number cannot serve that purpose.

CHOOSING THE AXIS. `--axis` is required unless `--axis-from` names a per-row
property file, in which case the axis is chosen by correlation and both
correlations are printed. There is deliberately no default: the informative axis
differs between systems, and a wrong guess produces a figure that looks right.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from icemotifs.relabel import axis_by_correlation, relabel_map


def main(argv=None):
    p = argparse.ArgumentParser(prog="icemotifs relabel")
    p.add_argument("--embedding", required=True, help="proj/embedding.npy")
    p.add_argument("--labels", required=True, help="motifs/labels.npy")
    p.add_argument("--out", required=True, help="directory for relabel.csv/.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--axis", type=int, choices=(0, 1),
                   help="0 = first projection axis, 1 = second")
    g.add_argument("--axis-from", metavar="NPY",
                   help="one value per row; the axis correlating with it is used")
    p.add_argument("--reverse", action="store_true",
                   help="rank descending, for a reversed colour ramp")
    a = p.parse_args(argv)

    emb = np.load(a.embedding)
    lab = np.load(a.labels)
    if len(emb) != len(lab):
        print(f"embedding has {len(emb)} rows but labels has {len(lab)}; these "
              "must come from the same run.", file=sys.stderr)
        return 2

    axis, r0, r1 = a.axis, None, None
    if a.axis_from:
        vals = np.load(a.axis_from)
        if len(vals) != len(emb):
            print(f"--axis-from has {len(vals)} values for {len(emb)} rows.",
                  file=sys.stderr)
            return 2
        axis, r0, r1 = axis_by_correlation(emb, lab, vals)
        print(f"  axis chosen by correlation: axis {axis}  "
              f"(r0 = {r0:+.3f}, r1 = {r1:+.3f})")
        if abs(abs(r0) - abs(r1)) < 0.05:
            print("  WARNING: the two correlations are within 0.05 of each "
                  "other, so this axis choice is close to arbitrary. State the "
                  "axis explicitly in any caption that depends on it.")

    rows = relabel_map(emb, lab, axis=axis, reverse=a.reverse)
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "relabel.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "relabel.json"), "w") as fh:
        json.dump({"axis": int(axis), "reverse": bool(a.reverse),
                   "n_motifs": len(rows), "n_rows": int(len(lab)),
                   "axis_from": a.axis_from,
                   "r_axis0": r0, "r_axis1": r1,
                   "note": "motif_relabel is a RANK along this axis, not an "
                           "identifier; motif is the vocabulary."}, fh, indent=2)

    lo, hi = rows[0], rows[-1]
    print(f"  {len(rows)} motifs ranked on axis {axis}"
          f"{' (reversed)' if a.reverse else ''}")
    print(f"    rank 1  -> motif {lo['motif']} (median {lo['median_axis']:+.2f},"
          f" {lo['percent_of_atoms']:.1f}% of atoms)")
    print(f"    rank {hi['motif_relabel']} -> motif {hi['motif']} "
          f"(median {hi['median_axis']:+.2f}, {hi['percent_of_atoms']:.1f}%)")
    print(f"  -> {csv_path}")
    print("  Colour by motif_relabel over the FULL range 1..n, pinned, not "
          "auto-ranged: the subset present in one frame is narrower than the "
          "vocabulary, and auto-ranging rescales it per frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
