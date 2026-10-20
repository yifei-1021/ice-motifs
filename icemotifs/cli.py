"""The `icemotifs` command: one entry point for the pipeline stages.

    icemotifs systems                         list the registered systems
    icemotifs check   --system S --dump F     measure d_nn BEFORE featurising
    icemotifs featurize --system S --input F --kind {dump,extxyz,filelist} \
                        --out DIR [--stride N]
    icemotifs train     --features DIR --system S --out DIR
    icemotifs encode    --features DIR --run DIR --out Z.npy
    icemotifs project   --z Z.npy --out DIR
    icemotifs cluster   --embedding E.npy --out DIR
    icemotifs relabel   --embedding E.npy --labels L.npy --out DIR --axis {0,1}
    icemotifs frames    --input F --labels L.npy --relabel R.csv --out DIR
    icemotifs all       --system S --input F [--kind K] --out DIR [--stride N]

`all` runs stages 1-5 in order, which is what `examples/<dataset>/scripts/`
does one script at a time. Going through the installed package instead means
the pipeline works from any directory once `pip install icemotifs` has run,
rather than only from inside the source tree with the parent directory on
PYTHONPATH.

CHECK d_nn FIRST. Every cutoff in the pipeline is a fixed multiple of the
first-neighbour distance, so a wrong `d_nn` silently rescales the SOAP radius,
the Steinhardt shells and the ring network at once. `icemotifs check` measures
it from your own trajectory and prints it next to the registry value.
"""
from __future__ import annotations

import argparse
import os
import sys

STAGES = {
    "check": ("icemotifs.pipeline.check",
              "measure d_nn BEFORE featurising (stage 0)"),
    "featurize": ("icemotifs.pipeline.featurize", "featurise a trajectory"),
    "train": ("icemotifs.pipeline.train_vae", "train the VAE"),
    "encode": ("icemotifs.pipeline.encode_run", "encode to the latent Z"),
    "project": ("icemotifs.pipeline.project", "PCA then UMAP; saves the reducer"),
    "cluster": ("icemotifs.pipeline.cluster_motifs", "HDBSCAN then k-means"),
    "relabel": ("icemotifs.pipeline.relabel_motifs",
                "rank motifs along a projection axis, for colouring"),
    "frames": ("icemotifs.pipeline.write_frames",
               "write motif-labelled extxyz for a renderer"),
}


def _run(stage, argv):
    import importlib
    mod = importlib.import_module(STAGES[stage][0])
    rc = mod.main(argv)
    return 0 if rc is None else int(rc)


def _systems():
    from icemotifs import systems as S
    names = [n for n in dir(S) if n.isupper() and isinstance(
        getattr(S, n), S.System)]
    print("registered systems (icemotifs/systems.py):\n")
    for n in sorted(names):
        print("  " + getattr(S, n).describe().replace("\n", "\n  "))
        print()
    print("To add your own, see tools/README.md -- it is one dataclass\n"
          "entry, and every cutoff is derived from d_nn.")
    return 0


def _all(argv):
    p = argparse.ArgumentParser(prog="icemotifs all")
    p.add_argument("--system", required=True)
    p.add_argument("--input", required=True,
                   help="trajectory: a LAMMPS dump, a multi-frame extxyz, or a "
                        "listing of per-frame extxyz files")
    p.add_argument("--kind", default="dump", choices=("dump", "extxyz", "filelist"),
                   help="which of the three the --input is (default: dump)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", default="20")
    p.add_argument("--skip-check", action="store_true",
                   help="skip the d_nn measurement (not recommended)")
    a = p.parse_args(argv)
    out = os.path.abspath(a.out)

    if not a.skip_check:
        print("=== 0. first-neighbour distance, before anything else")
        if a.kind == "dump":
            _run("check", ["--dump", a.input, "--system", a.system])
        else:
            print("    (d_nn measurement is implemented for --kind dump; for "
                  "extxyz/filelist pass --d-nn to the featurize stage, or "
                  "check it yourself against the registry value)")

    steps = [
        ("1. featurise", "featurize",
         ["--system", a.system, "--input", a.input, "--kind", a.kind,
          "--out", f"{out}/features", "--stride", str(a.stride)]),
        ("2. train the VAE", "train",
         ["--features", f"{out}/features", "--system", a.system,
          "--out", f"{out}/vae"]),
        ("3. encode", "encode",
         ["--features", f"{out}/features", "--run", f"{out}/vae",
          "--out", f"{out}/Z.npy"]),
        ("4. project", "project",
         ["--z", f"{out}/Z.npy", "--out", f"{out}/proj"]),
        ("5. cluster", "cluster",
         ["--embedding", f"{out}/proj/embedding.npy", "--out", f"{out}/motifs"]),
    ]
    for label, stage, args in steps:
        print(f"\n=== {label}")
        rc = _run(stage, args)
        if rc:
            print(f"\nstage '{stage}' exited {rc}; stopping here.", file=sys.stderr)
            return rc

    print(f"\ndone -> {out}")
    print("`all` stops here. The relabel stage is deliberately not included: it "
          "needs an axis, and the informative axis differs between systems, so "
          "running it with a default would produce a plausible wrong answer. "
          "See examples/README.md.")
    print("Report the DENSITY MAXIMA count and the OUTLIER FRACTION from")
    print(f"{out}/motifs/clustering.json -- not the motif total, which is the")
    print("maxima plus a chosen number of k-means cells.")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "systems":
        return _systems()
    if cmd == "all":
        return _all(rest)
    if cmd in STAGES:
        return _run(cmd, rest)
    print(f"unknown command {cmd!r}\n", file=sys.stderr)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
