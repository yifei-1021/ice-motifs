"""Everything specific to this dataset, in one place.

No path to any trajectory is recorded here -- the scripts take it as an
argument. OUT defaults to ./run beside this folder and can be overridden with
the OUT environment variable.
"""
import os
import sys

SYSTEM = "water-mw"        # registry entry; cutoffs are derived from its d_nn
KIND = "extxyz"            # dump | extxyz | filelist
# Overridable from the environment, because the published stride is wrong for
# the ten-frame sample in sample-data/: those frames are ALREADY spread across
# the published trajectory, so striding them again keeps almost nothing
# (STRIDE 500 over 10 frames keeps 1). Use STRIDE=1 on the sample.
STRIDE = int(os.environ.get("STRIDE", 500))   # keep every Nth frame when featurising
AXIS = 0              # projection axis the relabel ranking uses
DT_NS = 2.5            # simulation time per KEPT frame, for the x axis
TITLE = "Liquid-liquid transition: no crystal at either end"
N_ATOMS = None             # atoms per frame; inferred when it can be

# --- the published run, for the reproduction section of analysis.ipynb ------
# Provenance, not parameters: these four describe the run the panels in
# `reproduce-figures/` were drawn from. Stages 1-7 never read them, and a run
# of your own ignores them.
PUBLISHED_RUN_NAME = "Gartner-LDL-HDL/runs"     # for the EDIT-ME placeholder, nothing else
SOURCE = "192-235.0-3200-hi-water"     # its block prefix in that run's _pool/
# The published time axis: featurised frame i sits at
# (i + FRAME_OFFSET) * NS_PER_FRAME ns. This is NOT DT_NS. DT_NS is this
# example's own stride; the two differ wherever the stride does (ldl-hdl keeps
# every 500th frame, 2.5 ns apart, where the published run featurised every
# frame). Read from that run's REGENERATE-FIG-FROM-FUNCTIONS.ipynb.
NS_PER_FRAME = 0.004993757802746567
FRAME_OFFSET = 1
# The UMAP panels of a sheet share one frame, computed from the whole
# embedding. The published figures of this run do NOT clip its left edge at 0,
# so neither does this one; the value is read from that run's own
# REGENERATE-FIG-FROM-FUNCTIONS.ipynb rather than chosen here.
XLIM0 = False

# These frames are ALL-ATOM: 192 O + 384 H. The d_nn probe runs before the
# oxygens are selected, so it measures the O-H bond at 0.99 A and stops; and the
# set carries no per-atom energies. Both are why the production run passed
# --d-nn 2.8 --allow-missing-energy, which is what this dict reproduces.
FEATURIZE = {"d_nn": 2.8, "allow_missing_energy": True}   # extra stage-1 flags for this dataset

# This set carries no per-atom energies, so the energy target is all-NaN and
# fitting a scaler for it is a hard stop -- correctly, since a head trained
# against nothing would still report a loss. lambda_pred = 0 turns the property
# head OFF instead, which is what the production run did (its pipeline.log: "the
# property head is OFF. No energy or Voronoi prediction loss is optimised, and
# none should be reported for this run"). Say so in any methods section.
TRAIN = {"lambda_pred": 0}   # extra stage-2 flags for this dataset

# This file lives in scripts/, so OUT is the folder ABOVE it: output belongs to
# the example, not to the directory the code happens to sit in.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(HERE), "run"))


def need_input(argv):
    if len(argv) < 2:
        sys.exit(f"usage: {os.path.basename(argv[0])} /path/to/llt.extxyz (multi-frame extxyz)")
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
