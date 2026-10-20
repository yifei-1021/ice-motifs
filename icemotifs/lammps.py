"""
Read LAMMPS dump files into ASE frames the rest of the pipeline can featurise.

The published run read extended-xyz. The new mW and silicon trajectories are
LAMMPS dumps, and there are three things a plain `ase.io.read` does not do for
us:

  1. PER-ATOM ENERGY. The 281-column layout ends with potential energy per
     particle. In a dump that is a custom column (`c_pe` from
     `compute pe/atom`), which ASE does not attach to the frame. Without it the
     matrix silently ends up with a zero column and the VAE's second regression
     target is meaningless - so this module REFUSES to proceed rather than pad
     with zeros, unless the caller passes `allow_missing_energy=True`.

  2. ATOM ORDER. Dumps are not written in id order; LAMMPS emits them in
     whatever order the domain decomposition produced. Frame-to-frame
     comparisons (transition matrices, motif lifetimes) assume row i is the
     same particle in every frame, so every frame is sorted by id here. This is
     the single most common source of silently wrong transition matrices.

  3. ELEMENT. A dump has integer types, not symbols. pyscal3 and DScribe both
     key off species, so the type -> symbol map comes from the System.

Scaled coordinates (`xs ys zs`) and triclinic boxes are both handled.

This module is a reader, with no command line of its own: the d_nn report that
used to live here is stage 0, `icemotifs check`
(`icemotifs/pipeline/check.py`).
"""

from __future__ import annotations

import gzip
import json
import os
import io

import numpy as np


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def _box_to_cell(bounds, tilt):
    """LAMMPS bounds (+ optional xy xz yz) -> a 3x3 cell matrix."""
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    xy, xz, yz = tilt if tilt is not None else (0.0, 0.0, 0.0)
    # LAMMPS writes bounding-box bounds for triclinic cells; undo that
    if tilt is not None:
        xlo -= min(0.0, xy, xz, xy + xz)
        xhi -= max(0.0, xy, xz, xy + xz)
        ylo -= min(0.0, yz)
        yhi -= max(0.0, yz)
    return np.array([[xhi - xlo, 0.0, 0.0],
                     [xy, yhi - ylo, 0.0],
                     [xz, yz, zhi - zlo]], dtype=float), np.array([xlo, ylo, zlo])


def frame_offsets(path, cache=None):
    """Byte offset of every `ITEM: TIMESTEP` in a dump, as a list.

    WHY THIS EXISTS. `iter_dump` streams from the start of the file, so a job
    that featurises the trajectory in W windows re-parses every earlier window
    to reach its own: window k reads k*window frames it then throws away. For
    70 windows that is ~35x the necessary I/O, on a shared filesystem, for a
    multi-GB dump. One sequential scan builds the index; after that each window
    seeks straight to its first frame.

    Pass `cache` (a path) to keep the index next to the features, so the scan
    happens once per trajectory rather than once per window.
    """
    if cache and os.path.exists(cache):
        try:
            with open(cache) as fh:
                d = json.load(fh)
            if d.get("size") == os.path.getsize(path):
                return d["offsets"]
        except Exception:
            pass

    offsets, pos = [], 0
    with _open(path) as fh:
        for line in iter(fh.readline, ""):
            if line.startswith("ITEM: TIMESTEP"):
                offsets.append(pos)
            pos += len(line)

    if cache:
        # Per-process scratch name. Windows run concurrently and every one of
        # them may reach this line at the same moment; a shared "<cache>.tmp"
        # means one window renames the file out from under another, whose
        # os.replace then raises FileNotFoundError. And an index that cannot be
        # cached is a lost optimisation, not a lost result -- never fatal.
        tmp = f"{cache}.tmp.{os.getpid()}"
        try:
            d = os.path.dirname(os.path.abspath(cache))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(tmp, "w") as fh:
                json.dump({"size": os.path.getsize(path),
                           "offsets": offsets}, fh)
            os.replace(tmp, cache)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return offsets


# Every per-atom quantity LAMMPS can dump under a FIXED name. Anything in a
# dump that is not one of these is a compute/fix/variable the user named, and
# is therefore a candidate for the per-atom energy.
_RESERVED_DUMP_COLS = {
    "id", "mol", "proc", "procp1", "type", "element", "mass",
    "x", "y", "z", "xs", "ys", "zs", "xu", "yu", "zu",
    "xsu", "ysu", "zsu", "ix", "iy", "iz",
    "vx", "vy", "vz", "fx", "fy", "fz",
    "q", "mux", "muy", "muz", "mu",
    "radius", "diameter", "omegax", "omegay", "omegaz",
    "angmomx", "angmomy", "angmomz", "tqx", "tqy", "tqz",
}

# names people actually give `compute pe/atom`
_PE_NAME_HINTS = ("pe", "epot", "eng", "energy", "poteng")

# print the 'guessed the energy column' notice once per file, not once per frame
_announced = {}


def resolve_energy_column(cols, declared=None):
    """Decide which dump column holds the per-atom potential energy.

    Returns (name_or_None, how) where `how` is one of 'declared', 'named',
    'sole-compute' or 'none'. Kept separate from iter_dump so featurize.py can
    report the decision once per trajectory instead of once per frame.

    The three cases, in order:

      declared     --energy-column, or the system registry's energy_column.
                   Never guessed at, never overridden.
      named        a c_*/v_*/f_* column whose name contains pe, epot, eng, ...
                   (`c_pe`, `c_peatom`, `v_epot`).
      sole-compute EXACTLY ONE non-standard column exists, e.g. `c_1`. LAMMPS
                   lets you name a compute anything, and `compute 1 all
                   pe/atom` dumped as `c_1` is common. Using it is a guess, so
                   iter_dump additionally checks the values are negative --
                   per-atom potential energy is, for both mW and SW-Si, while
                   a coordination number or q6 that happened to be dumped as
                   `c_1` is not. Both the guess and the check are announced.
    """
    if declared and declared in cols:
        return declared, "declared"
    named = [c for c in cols
             if c[:2] in ("c_", "v_", "f_")
             and any(h in c[2:].lower() for h in _PE_NAME_HINTS)]
    if named:
        return named[0], "named"
    extra = [c for c in cols if c not in _RESERVED_DUMP_COLS]
    if len(extra) == 1 and extra[0][:2] in ("c_", "v_", "f_"):
        return extra[0], "sole-compute"
    return None, "none"


def iter_dump(path, system, *, allow_missing_energy=False, max_frames=None,
              start_frame=0, offsets=None, energy_column=None,
              stop_frame=None):
    """Yield (ase.Atoms, energy_array) per frame, sorted by atom id.

    `start_frame` together with `offsets` (from frame_offsets) seeks straight to
    that frame instead of streaming from the beginning. Frame numbering is still
    absolute: the first frame yielded is `start_frame`.

    `stop_frame` is that absolute index, exclusive, where the generator stops
    reading. The caller used to just stop consuming, which is not the same
    thing: the generator had already parsed one frame past its window, so a
    single malformed frame took down the window BEFORE it as well as its own.
    With stop_frame the damage stays in one window, and each window reads
    exactly the bytes it needs.
    """
    from ase import Atoms

    nframe = 0
    # absolute index of the frame about to be read. Without an index nframe
    # counts skipped frames too, so it IS the absolute index; with one we seek
    # and count from start_frame.
    base = start_frame if offsets is not None else 0
    with _open(path) as fh:
        if start_frame and offsets is not None:
            if start_frame >= len(offsets):
                return
            fh.seek(offsets[start_frame])
        # Without an index there is nothing safe to seek to: in text mode
        # tell() returns an opaque cookie, so byte arithmetic on it is invalid.
        # Parse from the start and simply do not yield frames below
        # start_frame; that is no slower than the old behaviour, and the
        # offsets path above is the fast one.
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            if stop_frame is not None and base + nframe >= stop_frame:
                return                       # do not even parse this frame
            fh.readline()                                    # the timestep
            assert fh.readline().startswith("ITEM: NUMBER OF ATOMS")
            natoms = int(fh.readline())

            hdr = fh.readline()
            assert hdr.startswith("ITEM: BOX BOUNDS"), hdr
            tilt = None
            if "xy xz yz" in hdr:
                tilt = []
                bounds = []
                for _ in range(3):
                    parts = fh.readline().split()
                    bounds.append((float(parts[0]), float(parts[1])))
                    tilt.append(float(parts[2]))
                tilt = (tilt[0], tilt[1], tilt[2])
            else:
                bounds = [tuple(float(x) for x in fh.readline().split()[:2])
                          for _ in range(3)]
            cell, origin = _box_to_cell(bounds, tilt)

            cols = fh.readline().split()[2:]                 # ITEM: ATOMS ...
            raw = np.array([fh.readline().split() for _ in range(natoms)],
                           dtype=float)
            ix = {c: i for i, c in enumerate(cols)}

            if "id" in ix:
                raw = raw[np.argsort(raw[:, ix["id"]])]

            if {"x", "y", "z"} <= set(ix):
                pos = raw[:, [ix["x"], ix["y"], ix["z"]]] - origin
            elif {"xs", "ys", "zs"} <= set(ix):
                frac = raw[:, [ix["xs"], ix["ys"], ix["zs"]]]
                pos = frac @ cell
            elif {"xu", "yu", "zu"} <= set(ix):
                pos = raw[:, [ix["xu"], ix["yu"], ix["zu"]]] - origin
            else:
                raise SystemExit(f"no position columns in dump: {cols}")

            if "type" in ix:
                types = raw[:, ix["type"]].astype(int)
                symbols = [system.type_map.get(t, system.symbol) for t in types]
            else:
                symbols = [system.symbol] * natoms

            # An explicitly requested column that is not in the file is a typo,
            # not an invitation to guess. The registry default (c_pe) missing is
            # normal -- that is the whole reason the other tiers exist.
            if energy_column and energy_column not in ix:
                raise SystemExit(
                    f"--energy-column {energy_column!r} is not in {path}.\n"
                    f"  columns present: {cols}")
            ecol, how = resolve_energy_column(
                cols, declared=energy_column or system.energy_column)
            if ecol is not None:
                energy = raw[:, ix[ecol]].astype(float)
                if how == "sole-compute" and not _announced.get(path):
                    _announced[path] = True
                    mean = float(np.mean(energy))
                    print(f"  energy column: guessed {ecol!r} -- the only "
                          f"non-standard column in {cols}", flush=True)
                    if not np.isfinite(mean) or mean >= 0.0:
                        raise SystemExit(
                            f"  ...but its mean is {mean:.4g}, and per-atom "
                            f"potential energy is negative for both mW and "
                            f"SW-Si.\n"
                            f"  {ecol!r} is probably not energy. Say which "
                            f"column it is with --energy-column, or pass "
                            f"--allow-missing-energy knowing the energy target "
                            f"will be zeros."
                        )
                    print(f"  mean {mean:.4g} (dump units) -- negative, "
                          f"consistent with per-atom PE", flush=True)
            elif allow_missing_energy:
                energy = np.zeros(natoms)
            else:
                raise SystemExit(
                    f"no per-atom energy column in {path}.\n"
                    f"  columns present: {cols}\n"
                    f"  If one of these IS the energy under another name (e.g.\n"
                    f"  `compute 1 all pe/atom` dumped as c_1), name it:\n"
                    f"      --energy-column <name>\n"
                    f"  Otherwise add `compute pe all pe/atom` and dump it as\n"
                    f"  c_pe, or pass --allow-missing-energy to fill the column\n"
                    f"  with zeros -- but then the VAE's energy regression\n"
                    f"  target is meaningless and the 281-column layout is a lie."
                )

            if start_frame and offsets is None and nframe + 0 < start_frame:
                nframe += 1                            # parsed, deliberately
                continue                               # not yielded
            atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)
            atoms.wrap()
            yield atoms, energy

            nframe += 1
            if max_frames and nframe >= max_frames:
                return


def count_frames(path):
    n = 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                n += 1
    return n
