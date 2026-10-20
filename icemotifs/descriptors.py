"""
The descriptors of stage 1: an extxyz trajectory to the 281-column matrix.

The stage that calls this is `icemotifs featurize`
(`icemotifs/pipeline/featurize.py`), which adds the d_nn check, the shard
layout and the dump/extxyz/filelist readers. Use these functions directly
to featurise one file in process.

Produces, per atom per frame:
    27 averaged Steinhardt (pyscal3, Lechner-Dellago) + 252 SOAP (DScribe)
    + Voronoi cell volume + potential energy

Differences from the original scripts, all deliberate:

  * the load_module() bootstrap is gone. It hardcoded one machine's conda
    prefix and was redundant anyway - a plain `import dscribe` followed
    immediately.
  * `frame.todict()["energy"]` is replaced. On ASE >= 3.23 `todict()` returns only
    cell/numbers/pbc/positions, so that call raises KeyError. See `_frame_energy`.
  * SOAP `sigma` and `rbf` are now explicit rather than relying on DScribe defaults.
    The published run used the defaults, which are sigma=1.0 and rbf='gto' - the
    values set here - but they were never written down.
  * no os.chdir; input and output are arguments.
"""

from __future__ import annotations


import numpy as np

from . import columns as C


# ----------------------------------------------------------------------------- Steinhardt + Voronoi

def _pyscal_system(frame):
    import pyscal3
    from pyscal3 import System

    d = frame.todict()
    s = System()
    s.box = d["cell"]
    s.atoms = pyscal3.Atoms({"positions": d["positions"], "species": d["numbers"]})
    return s


def steinhardt(frame, cutoffs=C.STEIN_CUTOFFS, ls=C.STEIN_L):
    """Averaged (Lechner-Dellago) Steinhardt parameters, one block per cutoff.

    pyscal3's `averaged=True` is the Lechner-Dellago form: it averages the complex
    Q_lm over the neighbourhood *including the central atom* and takes the modulus
    afterwards. Verified numerically to 3e-16 against a direct implementation.
    Note SI Eq. (S9) as printed describes a different quantity - the code is right.

    Ghost atoms: when the cell is small relative to the cutoff, pyscal3 replicates it
    and returns one value per real *and* ghost atom (e.g. 512 for a 64-atom cell at
    r_cut 6 A). The original code transposed that straight into the feature matrix,
    so it would have emitted rows for ghosts. It never showed up in practice because
    the published cell is 160 A and needs no replication at these cutoffs - but it
    silently corrupts any smaller system. Here the ghost mask is always applied.
    """
    out = []
    for rc in cutoffs:
        s = _pyscal_system(frame)
        s.find.neighbors(method="cutoff", cutoff=rc)
        q = np.asarray(s.calculate.steinhardt_parameter(list(ls), averaged=True))
        ghost = np.asarray(s.atoms["ghost"], dtype=bool)
        if q.shape[1] == ghost.size and ghost.any():
            q = q[:, ~ghost]
        out.extend(q)
    return np.asarray(out, dtype=np.float32).T          # (n_atoms, 27)


def _is_triclinic(cell, tol=1e-8):
    cell = np.asarray(cell, dtype=float)
    return not np.allclose(cell, np.diag(np.diag(cell)), atol=tol)


def voronoi_volume_periodic(frame, margin=None):
    """Per-atom Voronoi volume for ANY cell, orthorhombic or tilted.

    Built directly on scipy's Voronoi rather than pyscal3, because pyscal3's
    triclinic branch is broken: identify.find_neighbors() calls `operations.repeat`
    without importing `operations`, and the line after it calls
    `System.embed_in_cubic_box`, which does not exist. Both fire only when
    `system.triclinic` is true, which is why orthorhombic dumps never hit it.

    Periodicity is handled by surrounding the cell with enough image points that
    every real atom's Voronoi cell is closed and identical to the infinite-lattice
    one. `margin` is that padding in Angstrom; the default, three mean particle
    spacings, is far beyond the reach of any Voronoi neighbour in a condensed phase.
    The returned volumes must sum to the cell volume -- that identity is asserted,
    so a margin that was somehow too small becomes an error and not a quiet bias.
    """
    from scipy.spatial import ConvexHull, Voronoi

    cell = np.asarray(frame.get_cell(), dtype=float)
    pos = np.asarray(frame.get_positions(), dtype=float)
    n = len(pos)
    vol = abs(float(np.linalg.det(cell)))
    if margin is None:
        margin = 3.0 * (vol / max(n, 1)) ** (1.0 / 3.0)

    # wrap into the cell, so the padding shell is symmetric around it
    frac = np.linalg.solve(cell.T, pos.T).T % 1.0
    pos = frac @ cell

    # perpendicular width of the cell along each lattice direction: how far one
    # image step actually moves you, which is what the margin has to be measured in
    widths = np.array([
        vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ])
    reps = np.ceil(margin / widths).astype(int)
    pad = margin / widths                       # margin in fractional units

    pts = [pos]
    for i in range(-reps[0], reps[0] + 1):
        for j in range(-reps[1], reps[1] + 1):
            for k in range(-reps[2], reps[2] + 1):
                if i == j == k == 0:
                    continue
                shift = np.array([i, j, k], dtype=float)
                f = frac + shift
                keep = np.all((f > -pad) & (f < 1.0 + pad), axis=1)
                if keep.any():
                    pts.append(pos[keep] + shift @ cell)
    allpts = np.vstack(pts)

    vor = Voronoi(allpts)
    out = np.empty(n, dtype=np.float64)
    for a in range(n):
        region = vor.regions[vor.point_region[a]]
        if not region or -1 in region:          # unbounded: margin was too small
            raise RuntimeError(
                f"atom {a} has an open Voronoi cell at margin {margin:.2f} A; "
                f"raise the margin")
        out[a] = ConvexHull(vor.vertices[region]).volume

    total = out.sum()
    if not np.isclose(total, vol, rtol=1e-6):
        raise RuntimeError(
            f"Voronoi volumes sum to {total:.6f} A^3 but the cell is "
            f"{vol:.6f} A^3 -- the tessellation does not tile the box")
    return out.astype(np.float32).reshape(-1, 1)


def voronoi_volume(frame):
    """Per-atom Voronoi cell volume.

    Orthorhombic cells go through pyscal3, which is the published path and the
    faster one. Tilted cells go through `voronoi_volume_periodic`, because
    pyscal3's triclinic branch raises (see that function). The two agree to
    float32 on cells where both run.

    `find.neighbors(method="voronoi")` already populates the per-atom volume, so the
    `calculate.voronoi_vector()` call the original made is unnecessary here - and on
    pyscal3 3.3.2 + NumPy 2 it raises ("inhomogeneous shape"), because it tries to
    build a ragged edge-length array. Only the Voronoi *vector* needs it, and the
    281-column layout does not use the vector.

    Sanity-checked: the volumes returned this way sum exactly to the cell volume.
    """
    if _is_triclinic(frame.get_cell()):
        return voronoi_volume_periodic(frame)
    s = _pyscal_system(frame)
    s.find.neighbors(method="voronoi")
    return np.asarray(s.atoms.voronoi.volume, dtype=np.float32).reshape(-1, 1)


# ----------------------------------------------------------------------------- SOAP

def soap(frame, r_cut=6.0, n_max=C.SOAP_N_MAX, l_max=C.SOAP_L_MAX,
         sigma=1.0, rbf="gto", species=("O",), periodic=True):
    """Per-atom SOAP power spectrum.

    r_cut=6.0 is the published value (Methods; justified by Fig. S1). 3.2 and 4.8 A
    variants of the feature matrix also exist - they are NOT interchangeable, and a
    model trained on one gives ~2000x worse SOAP reconstruction on another.
    """
    from dscribe.descriptors import SOAP

    desc = SOAP(species=list(species), periodic=periodic, r_cut=r_cut,
                n_max=n_max, l_max=l_max, sigma=sigma, rbf=rbf)
    return np.asarray(desc.create(frame), dtype=np.float32)


# ----------------------------------------------------------------------------- energy

def _frame_energy(frame, n_atoms, energy_column=None):
    """Per-atom potential energy, robust across ASE versions.

    The original `frame.todict()["energy"]` no longer works: ASE 3.29's todict()
    returns only cell/numbers/pbc/positions. Try, in order, a per-atom array, then
    a total energy broadcast over atoms, then NaN.

    `energies` is a *reserved* name in extxyz: ASE writes it out but on read routes it
    to `frame.calc.results`, not to `frame.arrays`. Checking only `arrays` therefore
    silently yields NaN for exactly the column the VAE regresses on. Check the
    calculator first; a custom column name such as LAMMPS' `c_pe` does land in
    `arrays` and is checked too.
    """
    if energy_column:
        # an explicitly named column wins, and a name that is not there is a
        # typo rather than an invitation to fall through to the guesses
        for src in (frame.arrays, frame.info):
            if energy_column in src:
                a = np.asarray(src[energy_column], dtype=np.float32).ravel()
                if a.size == n_atoms:
                    return a.reshape(-1, 1)
        raise SystemExit(
            f"--energy-column {energy_column!r} is not a per-atom array in "
            f"this frame. Present: {sorted(frame.arrays)}")

    if frame.calc is not None:
        for key in ("energies", "energy"):
            v = frame.calc.results.get(key)
            if v is not None:
                a = np.asarray(v, dtype=np.float32).ravel()
                if a.size == n_atoms:
                    return a.reshape(-1, 1)
                if a.size == 1:
                    return np.full((n_atoms, 1), float(a[0]) / n_atoms, dtype=np.float32)

    # Named columns declared in the extxyz Properties= string land in
    # frame.arrays under exactly that name. `pe` is what the supercooled-water
    # sets use -- Properties=species:S:1:pos:R:3:pe:R:1:... -- and leaving it
    # out of this list is why those frames read as NaN energy for every atom.
    # Ordered: the unambiguous names first, then anything whose name looks like
    # a potential energy and has one value per atom.
    for key in ("energies", "energy", "pe", "c_pe", "potential_energy",
                "epot", "pe_atom", "peatom", "eng"):
        if key in frame.arrays:
            a = np.asarray(frame.arrays[key], dtype=np.float32).ravel()
            if a.size == n_atoms:
                return a.reshape(-1, 1)
    _RESERVED = {"numbers", "positions", "species", "symbols", "momenta",
                 "forces", "tags", "masses", "initial_charges",
                 "initial_magmoms", "move_mask"}
    for key, v in frame.arrays.items():
        if key in _RESERVED:
            continue
        low = key.lower()
        if not any(h in low for h in ("pe", "epot", "eng", "energy")):
            continue
        a = np.asarray(v, dtype=np.float32).ravel()
        if a.size == n_atoms:
            return a.reshape(-1, 1)
    for key in ("energies", "c_pe"):
        if key in frame.info:
            a = np.asarray(frame.info[key], dtype=np.float32).ravel()
            if a.size == n_atoms:
                return a.reshape(-1, 1)
    try:
        total = float(frame.get_potential_energy())
        return np.full((n_atoms, 1), total / n_atoms, dtype=np.float32)
    except Exception:
        pass
    if "energy" in frame.info:
        return np.full((n_atoms, 1), float(frame.info["energy"]) / n_atoms, dtype=np.float32)
    return np.full((n_atoms, 1), np.nan, dtype=np.float32)


# ----------------------------------------------------------------------------- driver

def featurize_frame(frame, r_cut=6.0, **soap_kw):
    n = len(frame)
    blocks = [steinhardt(frame), soap(frame, r_cut=r_cut, **soap_kw),
              voronoi_volume(frame), _frame_energy(frame, n)]
    row = np.hstack(blocks).astype(np.float32)
    return C.LAYOUT_281.check(row)


def featurize_file(path, r_cut=6.0, stride=1, index=":", verbose=True, **soap_kw):
    from ase.io import read

    frames = read(path, format="extxyz", index=index)
    if not isinstance(frames, list):
        frames = [frames]
    frames = frames[::stride]
    out = []
    for i, f in enumerate(frames):
        out.append(featurize_frame(f, r_cut=r_cut, **soap_kw))
        if verbose:
            print(f"  frame {i + 1}/{len(frames)}: {len(f)} atoms", flush=True)
    return np.vstack(out)
