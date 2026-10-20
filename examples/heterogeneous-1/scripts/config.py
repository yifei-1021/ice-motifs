"""Everything specific to this dataset, in one place.

No path to any trajectory is recorded here -- the scripts take it as an
argument. OUT defaults to ./run beside this folder and can be overridden with
the OUT environment variable.
"""
import os
import sys

SYSTEM = "water-mlbop"        # registry entry; cutoffs are derived from its d_nn
KIND = "dump"            # dump | extxyz | filelist
# Overridable from the environment, because the published stride is wrong for
# the ten-frame sample in sample-data/: those frames are ALREADY spread across
# the published trajectory, so striding them again keeps almost nothing
# (STRIDE 500 over 10 frames keeps 1). Use STRIDE=1 on the sample.
STRIDE = int(os.environ.get("STRIDE", 2))   # keep every Nth frame when featurising
AXIS = 1              # projection axis the relabel ranking uses
DT_NS = 0.2            # simulation time per KEPT frame, for the x axis
TITLE = "Heterogeneous nucleation at liquid/polymorph interface 1"
N_ATOMS = None             # atoms per frame; inferred when it can be

# --- the published run, for the reproduction section of analysis.ipynb ------
# Provenance, not parameters: these four describe the run the panels in
# `reproduce-figures/` were drawn from. Stages 1-7 never read them, and a run
# of your own ignores them.
# The four heterogeneous systems share ONE clustering, so this folder takes
# its own trajectory's rows out of the pooled run rather than having a run of
# its own.
PUBLISHED_RUN_NAME = "heterogenous/runs"     # for the EDIT-ME placeholder, nothing else
SOURCE = "8154387"                     # its block prefix in that run's _pool/
# The published time axis: featurised frame i sits at
# (i + FRAME_OFFSET) * NS_PER_FRAME ns. This is NOT DT_NS. DT_NS is this
# example's own stride; the two differ wherever the stride does (ldl-hdl keeps
# every 500th frame, 2.5 ns apart, where the published run featurised every
# frame). Read from that run's REGENERATE-FIG-FROM-FUNCTIONS.ipynb.
NS_PER_FRAME = 0.2
FRAME_OFFSET = 1
# The UMAP panels of a sheet share one frame, computed from the whole
# embedding. The published figures of this run clip its left edge at 0, so this
# one does too; the value is read from that run's own
# REGENERATE-FIG-FROM-FUNCTIONS.ipynb rather than chosen here.
XLIM0 = True

# What the production featurisation of this trajectory passed. The energy
# column in these dumps is `c_1`; naming it means a renamed column is an error
# rather than a different guess. (SYSTEM here is `water-mlbop`, which declares
# no energy column at all; the production run used `water-mw`, which declares
# `c_pe`. Pinning d_nn makes every cutoff agree either way.)
FEATURIZE = {"d_nn": 2.76, "energy_column": "c_1"}   # extra stage-1 flags for this dataset

TRAIN = {}                   # extra stage-2 flags for this dataset

# This file lives in scripts/, so OUT is the folder ABOVE it: output belongs to
# the example, not to the directory the code happens to sit in.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(HERE), "run"))


def need_input(argv):
    if len(argv) < 2:
        sys.exit(f"usage: {os.path.basename(argv[0])} /path/to/interface.dump (LAMMPS dump)")
    if not os.path.exists(argv[1]):
        sys.exit(f"input not found: {argv[1]}")
    return argv[1]


def flags(kw):
    """{"d_nn": 2.8, "allow_missing_energy": True} ->
    ["--d-nn", "2.8", "--allow-missing-energy"].

    Shared by the scripts and the notebook so a flag is spelled once. Only what
    is passed appears: every default stays in the stage's own argument parser,
    where a copy here would quietly go stale.
    """
    argv = []
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is None or v is False:
            continue
        else:
            argv += [flag, str(v)]
    return argv


def need_stage(path, script):
    """Stop unless `script` actually produced output.

    Checking that the directory EXISTS is not enough: a stage that fails partway
    leaves an empty directory behind, and the next stage then fails with its own
    confusing message instead of the real one. So this requires content.
    """
    if not os.path.isdir(path) or not os.listdir(path):
        sys.exit(f"{path} is missing or empty -- run {script} first "
                 "(and check it finished: an empty directory means it did not)")


def n_atoms_per_frame(out, n_labels):
    """Atoms per frame, from N_ATOMS or from the blocks stage 1 wrote.

    Stage 1 writes one `frame_%06d.npy` per frame, so the row count of a block
    IS the atom count. An earlier version of this function looked for a
    `features/features.json` that the featuriser has never written, so it
    always returned None and the population plot was silently skipped on every
    run.
    """
    if N_ATOMS:
        return N_ATOMS
    from icemotifs.data import frames_in
    rows = frames_in(os.path.join(out, "features"))
    if not rows:
        return None
    if len(set(rows)) != 1:
        print(f"  frames hold differing atom counts ({sorted(set(rows))[:4]}); "
              "a population series needs a constant count, so it is skipped")
        return None
    if sum(rows) != n_labels:
        print(f"  {sum(rows)} feature rows against {n_labels} labels -- not the "
              "same run, so atoms-per-frame is not trustworthy here")
        return None
    return rows[0]


def load_properties(out, n_rows):
    """Per-atom properties for the tables and figures, read from the features.

    The featuriser does not write one array per property. It writes the whole
    descriptor matrix per frame, with the regression targets in its last
    columns, so the columns are pulled out here (see icemotifs.columns for the
    layout):

        rho     Voronoi cell volume, converted to a local density
        energy  per-atom potential energy, when the trajectory carried one
        q6      qbar_6 at the FIRST Steinhardt cutoff

    Blocks are read in sorted filename order, which is frame order -- the same
    order the labels are in, and the only order in which row i means the same
    atom in both.

    Returns {} rather than raising when the features are absent: the analysis
    should degrade to the columns that exist instead of failing.
    """
    import numpy as np
    from icemotifs import columns as C
    from icemotifs.motifs import voronoi_volume_to_density
    from icemotifs.systems import get as get_system

    feat = os.path.join(out, "features")
    files = sorted(f for f in os.listdir(feat)) if os.path.isdir(feat) else []
    files = [f for f in files if f.endswith(".npy")]
    if not files:
        print(f"  no feature blocks under {feat}; tables and property panels "
              "will be skipped. See the folder README.")
        return {}

    layout = C.infer(int(np.load(os.path.join(feat, files[0]),
                                 mmap_mode="r").shape[1]))
    # The Steinhardt l values run (4, 5, 6, 8, ...) once per cutoff, so qbar_6
    # is the third column of the first cutoff's nine.
    cols = {"q6": C.STEIN_L.index(6)}
    for i, name in enumerate(layout.targets):
        col = layout.targets_slice.start + i
        if name == "voronoi_volume":
            cols["rho"] = col
        elif name == "energy":
            cols["energy"] = col

    found = {k: [] for k in cols}
    for f in files:
        block = np.load(os.path.join(feat, f), mmap_mode="r")
        for k, c in cols.items():
            found[k].append(np.asarray(block[:, c], dtype=np.float64))
    found = {k: np.concatenate(v) for k, v in found.items()}

    if "rho" in found:
        found["rho"] = voronoi_volume_to_density(found["rho"],
                                                 get_system(SYSTEM).mass)
    for k in list(found):
        v = found[k]
        # 1000.0 is the published matrix's "not a measurement" marker for the
        # regression targets (reference phases, excluded from the loss); a
        # trajectory with no per-atom energy leaves NaN instead. Either way it
        # is not a value to plot.
        if k in ("rho", "energy"):
            v = np.where(v == 1000.0, np.nan, v)
        if len(v) != n_rows:
            print(f"  {k}: {len(v)} feature rows for {n_rows} labels; dropped")
            del found[k]
        elif not np.isfinite(v).any():
            print(f"  {k}: no finite values in this trajectory; dropped")
            del found[k]
        else:
            found[k] = v
    if not found:
        print(f"  no usable property columns in {feat}; tables and property "
              "panels will be skipped. See the folder README.")
    return found
