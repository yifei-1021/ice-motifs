#!/usr/bin/env python3
"""Test suite for the S9 pipeline. Runs without any real trajectory.

    python tools/test_pipeline.py          # prints PASS/FAIL, exits with the count
    python -m pytest tools/test_pipeline.py

Covers the things that are easy to get silently wrong: cutoff rescaling, the
LAMMPS reader's id ordering and energy handling, the layout width, the
two-stage clustering contract, and the reducer-size problem that stopped the
published reducer being saved.
"""
import json
import os
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from icemotifs import (cluster, columns as C, descriptors as D,   # noqa: E402
                       lammps, systems as S)

sys.path.insert(0, HERE)
from make_synthetic import dump as make_dump                        # noqa: E402

CHECKS = []


def check(name):
    """Register a check under a readable name.

    The function itself is named `test_...` so pytest collects it; this only
    records the pair, so importing the module runs nothing. Running the file
    directly executes them in definition order (see __main__ at the bottom).
    """
    def deco(fn):
        fn.check_name = name
        CHECKS.append(fn)
        return fn
    return deco


TMP = tempfile.mkdtemp(prefix="icemotifs-test-")
DUMP = os.path.join(TMP, "syn.lammpstrj")
N_ATOMS, N_FRAMES = make_dump(DUMP, n_frames=4, nx=5)
print(f"synthetic dump: {N_FRAMES} frames x {N_ATOMS} atoms\n")


@check("water cutoffs reproduce the published values exactly")
def test_water_cutoffs_reproduce_the_published_values_exactly():
    w = S.WATER_MLBOP
    assert w.soap_rcut == 6.0, w.soap_rcut
    assert w.stein_cutoffs == (3.2, 4.8, 6.0), w.stein_cutoffs
    assert w.ring_cutoff == 3.2, w.ring_cutoff
    assert w.coord_cutoff == 6.0, w.coord_cutoff


@check("silicon cutoffs are rescaled, not copied")
def test_silicon_cutoffs_are_rescaled_not_copied():
    si = S.SILICON_SW
    assert si.soap_rcut < 6.0, si.soap_rcut
    ratio = si.soap_rcut / S.WATER_MLBOP.soap_rcut
    assert abs(ratio - si.d_nn / S.WATER_D_NN) < 1e-6
    # every cutoff scales by the SAME factor
    for a, b in zip(si.stein_cutoffs, S.WATER_MLBOP.stein_cutoffs):
        assert abs(a / b - ratio) < 1e-4, (a, b)


@check("dump reader sorts by atom id")
def test_dump_reader_sorts_by_atom_id():
    prev = None
    for atoms, _e in lammps.iter_dump(DUMP, S.SILICON_SW, max_frames=2):
        p = atoms.get_positions()
        assert len(p) == N_ATOMS
        if prev is not None:
            # consecutive frames differ only by the noise, so the same row
            # must be the same particle: displacement stays small
            d = np.linalg.norm(p - prev, axis=1)
            assert np.median(d) < 3.0, f"median displacement {np.median(d):.2f} A"
        prev = p


@check("dump reader refuses to fabricate a missing energy column")
def test_dump_reader_refuses_to_fabricate_a_missing_energy_column():
    stripped = os.path.join(TMP, "noenergy.lammpstrj")
    with open(DUMP) as fi, open(stripped, "w") as fo:
        for line in fi:
            if line.startswith("ITEM: ATOMS"):
                fo.write("ITEM: ATOMS id type x y z\n")
            elif line and line[0].isdigit() and len(line.split()) == 6:
                fo.write(" ".join(line.split()[:5]) + "\n")
            else:
                fo.write(line)
    try:
        next(iter(lammps.iter_dump(stripped, S.SILICON_SW)))
    except SystemExit as exc:
        assert "per-atom energy" in str(exc)
        return
    raise AssertionError("should have refused")


def _rename_energy_col(newname, fname):
    """Copy DUMP with its c_pe column renamed. Real dumps do this: `compute 1
    all pe/atom` lands in the file as c_1, not c_pe."""
    out = os.path.join(TMP, fname)
    with open(DUMP) as fi, open(out, "w") as fo:
        for line in fi:
            if line.startswith("ITEM: ATOMS"):
                fo.write(line.replace("c_pe", newname))
            else:
                fo.write(line)
    return out


@check("an energy column under another name is found, named or declared")
def test_an_energy_column_under_another_name_is_found_named_or_decl():
    ref = next(iter(lammps.iter_dump(DUMP, S.SILICON_SW)))[1]

    # 1. the sole non-standard column is used, and the guess is announced
    c1 = _rename_energy_col("c_1", "c1.lammpstrj")
    lammps._announced.clear()
    name, how = lammps.resolve_energy_column(
        ["id", "type", "x", "y", "z", "c_1"], declared="c_pe")
    assert (name, how) == ("c_1", "sole-compute"), (name, how)
    got = next(iter(lammps.iter_dump(c1, S.SILICON_SW)))[1]
    assert np.allclose(got, ref)

    # 2. an explicit --energy-column beats everything
    got = next(iter(lammps.iter_dump(c1, S.SILICON_SW, energy_column="c_1")))[1]
    assert np.allclose(got, ref)

    # 3. a differently-spelled pe name is matched by hint, not by guessing
    for nm in ("c_peatom", "v_epot", "c_myeng"):
        n, how = lammps.resolve_energy_column(
            ["id", "type", "x", "y", "z", nm, "c_coord"], declared="c_pe")
        assert (n, how) == (nm, "named"), (nm, n, how)

    # 4. two anonymous computes is ambiguous -> refuse, do not pick one
    n, how = lammps.resolve_energy_column(
        ["id", "type", "x", "y", "z", "c_1", "c_2"], declared="c_pe")
    assert (n, how) == (None, "none"), (n, how)


@check("a guessed energy column is rejected when the values are not energies")
def test_a_guessed_energy_column_is_rejected_when_the_values_are_no():
    # c_1 holding a coordination number: positive, so not per-atom PE. The
    # guess must not go through silently.
    fake = os.path.join(TMP, "coord.lammpstrj")
    with open(DUMP) as fi, open(fake, "w") as fo:
        for line in fi:
            if line.startswith("ITEM: ATOMS"):
                fo.write(line.replace("c_pe", "c_1"))
            elif line and line[0].isdigit() and len(line.split()) == 6:
                p = line.split(); p[5] = "4.0"          # coordination, positive
                fo.write(" ".join(p) + "\n")
            else:
                fo.write(line)
    lammps._announced.clear()
    try:
        next(iter(lammps.iter_dump(fake, S.SILICON_SW)))
    except SystemExit as exc:
        assert "probably not energy" in str(exc), str(exc)
        return
    raise AssertionError("a positive-mean column was accepted as energy")


@check("Voronoi volumes tile the box, tilted cell or not")
def test_voronoi_volumes_tile_the_box_tilted_cell_or_not():
    from ase import Atoms
    rng = np.random.default_rng(7)

    # 1. orthorhombic: the scipy path must reproduce pyscal3 exactly, or the
    #    two branches would put different numbers in column 279 of one dataset.
    L = 24.0
    cell = np.eye(3) * L
    ortho = Atoms("O" * 400, positions=rng.random((400, 3)) @ cell,
                  cell=cell, pbc=True)
    a = D.voronoi_volume(ortho)                  # pyscal3
    b = D.voronoi_volume_periodic(ortho)         # scipy
    assert np.array_equal(a, b), np.abs(a - b).max()
    assert np.isclose(a.sum(), L ** 3, rtol=1e-6), (a.sum(), L ** 3)

    # 2. tilted: pyscal3 raises here (NameError: operations), so voronoi_volume
    #    must route around it rather than propagate the crash.
    tcell = np.array([[L, 0.0, 0.0], [2.0, L, 0.0], [1.0, -1.5, L]])
    tri = Atoms("O" * 400, positions=rng.random((400, 3)) @ tcell,
                cell=tcell, pbc=True)
    assert D._is_triclinic(tcell) and not D._is_triclinic(cell)
    v = D.voronoi_volume(tri)
    assert v.shape == (400, 1)
    assert np.isclose(v.sum(), abs(np.linalg.det(tcell)), rtol=1e-6)
    assert (v > 0).all()


@check("concurrent windows can all write the metadata without killing each other")
def test_concurrent_windows_can_all_write_the_metadata_without_kill():
    """The 11 Sep failure: ten windows, one shared "<file>.tmp", and whichever
    window renamed first made every other window's os.replace raise."""
    import multiprocessing as mp

    target = os.path.join(TMP, "race", "d_nn.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    class A:                                   # the bits _write_dnn reads
        dnn_file = target
        input = DUMP

    def worker(q):
        try:
            from icemotifs.pipeline import featurize as F
            for _ in range(40):
                F._write_dnn(A(), S.WATER_MW, measured=None, n=0, pinned=True)
            q.put(None)
        except Exception as exc:               # noqa: BLE001
            q.put(f"{type(exc).__name__}: {exc}")

    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(q,)) for _ in range(10)]
    for p_ in procs:
        p_.start()
    errs = [q.get() for _ in procs]
    for p_ in procs:
        p_.join()

    bad = [e for e in errs if e]
    assert not bad, bad
    with open(target) as fh:
        assert abs(json.load(fh)["d_nn_used"] - S.WATER_MW.d_nn) < 1e-12
    leftovers = [f for f in os.listdir(os.path.dirname(target)) if ".tmp" in f]
    assert not leftovers, leftovers


@check("filelist reader opens only the frames the stride keeps")
def test_filelist_reader_opens_only_the_frames_the_stride_keeps():
    """The N64 set is one 64,000-atom extxyz file PER FRAME at stride 100. The
    reader used to parse all 100 files between wanted frames and the caller
    discarded 99 -- which is why the d_nn probe looked hung."""
    import builtins
    import tempfile
    from ase import Atoms
    from ase.io import write

    from icemotifs.pipeline import featurize as Fz

    d = tempfile.mkdtemp(prefix="filelist-")
    rng = np.random.default_rng(0)
    for i in range(60):
        at = Atoms("O" * 20, positions=rng.random((20, 3)) * 10,
                   cell=np.eye(3) * 10, pbc=True)
        write(os.path.join(d, f"f{i:03d}.extxyz"), at, format="extxyz")
    listing = os.path.join(d, "list.txt")
    with open(listing, "w") as fh:
        fh.write("\n".join(f"f{i:03d}.extxyz" for i in range(60)))

    seen, real = [], builtins.open

    def spy(path, *args, **kw):
        if str(path).endswith(".extxyz"):
            seen.append(os.path.basename(str(path)))
        return real(path, *args, **kw)

    builtins.open = spy
    try:
        got = [k for k, _a, _e in Fz._iter_frames(
            "filelist", listing, None, root=d, keep={0, 20, 40}, stop=60)]
    finally:
        builtins.open = real

    assert got == [0, 20, 40], got
    assert len(set(seen)) == 3, sorted(set(seen))


@check("triclinic box bounds are converted to a cell")
def test_triclinic_box_bounds_are_converted_to_a_cell():
    tri = os.path.join(TMP, "tri.lammpstrj")
    with open(tri, "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n")
        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        f.write("0.0 10.0 2.0\n0.0 10.0 0.0\n0.0 10.0 0.0\n")
        f.write("ITEM: ATOMS id type x y z c_pe\n")
        f.write("1 1 1.0 1.0 1.0 -4.0\n2 1 2.0 2.0 2.0 -4.0\n")
    atoms, _ = next(iter(lammps.iter_dump(tri, S.SILICON_SW)))
    cell = np.asarray(atoms.get_cell())
    assert abs(cell[1, 0] - 2.0) < 1e-9, cell     # xy tilt survived
    assert cell[0, 1] == 0.0 and cell[0, 2] == 0.0


@check("featurised block has exactly 281 columns and one row per atom")
def test_featurised_block_has_exactly_281_columns_and_one_row_per_a():
    from icemotifs.pipeline import featurize as fz
    atoms, energy = next(iter(lammps.iter_dump(DUMP, S.SILICON_SW)))
    block = fz.featurize_frame(atoms, energy, S.SILICON_SW)
    assert block.shape == (len(atoms), C.LAYOUT_281.n_cols), block.shape
    assert np.isfinite(block[:, 279]).all(), "Voronoi volumes not finite"
    assert (block[:, 279] > 0).all(), "non-positive Voronoi volume"
    # volumes must sum to the cell volume
    v = abs(np.linalg.det(np.asarray(atoms.get_cell())))
    assert abs(block[:, 279].sum() - v) / v < 1e-3, \
        (block[:, 279].sum(), v)


@check("two-stage clustering: maxima + kmeans cells, outliers reported")
def test_two_stage_clustering_maxima_kmeans_cells_outliers_reported():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(c, 0.25, (800, 2))
                       for c in [(0, 0), (6, 0), (0, 6)]])
    # A diffuse continuum, deliberately below the density of the blobs: this is
    # the intermediate region the paper's transient motifs live in. Made too
    # dense it becomes a fourth maximum, which is itself the point -- what
    # counts as a "cluster" here is set by min_cluster_size.
    bridge = rng.uniform(-1.0, 7.0, (250, 2))
    E = np.vstack([blobs, bridge])
    part = cluster.two_stage(E, min_cluster_size=200, n_kmeans=4, seed=0)
    assert part.n_density_maxima == 3, part.n_density_maxima
    assert part.n_kmeans == 4
    assert part.n_motifs == 7
    assert 0.02 < part.outlier_fraction < 0.35, part.outlier_fraction
    assert set(part.counts) == set(range(7)), sorted(part.counts)


@check("empty intermediate region is reported, not crashed on")
def test_empty_intermediate_region_is_reported_not_crashed_on():
    rng = np.random.default_rng(1)
    E = np.vstack([rng.normal(c, 0.15, (600, 2)) for c in [(0, 0), (10, 0)]])
    part = cluster.two_stage(E, min_cluster_size=100, n_kmeans=500, seed=0)
    assert part.n_kmeans == 0
    assert part.n_motifs == part.n_density_maxima


@check("nearest-point assignment labels every row")
def test_nearest_point_assignment_labels_every_row():
    rng = np.random.default_rng(2)
    E = rng.normal(0, 1, (5000, 2))
    sub = E[::10]
    lab = rng.integers(0, 5, len(sub))
    out, _ = cluster.assign_rest(E, sub, lab)
    assert len(out) == len(E)
    assert set(np.unique(out)) <= set(np.unique(lab))
    # points that ARE in the subsample must keep their own label
    assert (out[::10] == lab).all()


@check("measure_d_nn finds the FIRST peak, not the tallest")
def test_measure_d_nn_finds_the_first_peak_not_the_tallest():
    atoms, _ = next(iter(lammps.iter_dump(DUMP, S.SILICON_SW)))
    d = S.measure_d_nn(atoms, rmax=5.0)
    assert abs(d - 2.35) < 0.15, f"got {d}, diamond first shell is 2.35 A"


@check("reducer size scales with the FIT set, not the data")
def test_reducer_size_scales_with_the_fit_set_not_the_data():
    import pickle
    try:
        import umap
    except ImportError:
        print("      (umap-learn absent, skipped)")
        return
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (4000, 8))
    sizes = {}
    for n in (500, 2000):
        r = umap.UMAP(n_neighbors=10, random_state=0).fit(X[:n])
        sizes[n] = len(pickle.dumps(r))
    assert sizes[2000] > sizes[500], sizes
    print(f"      fit 500 -> {sizes[500]/1e3:.0f} kB, "
          f"fit 2000 -> {sizes[2000]/1e3:.0f} kB")



@check("sp_rings is exact on a lattice where the answer is known")
def test_sp_rings_is_exact_on_a_lattice_where_the_answer_is_known():
    import networkx as nx
    from icemotifs.rings import sp_rings
    from collections import Counter
    # diamond: every atom in exactly 12 six-rings -> 2N six-rings, none shorter
    a = 5.43
    b = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],
                  [.25,.25,.25],[.25,.75,.75],[.75,.25,.75],[.75,.75,.25]])
    c = np.array([[i,j,k] for i in range(3) for j in range(3) for k in range(3)])
    pos = ((c[:,None,:]+b[None,:,:]).reshape(-1,3))*a
    L = 3*a
    d = pos[:,None,:]-pos[None,:,:]; d -= L*np.rint(d/L)
    r = np.linalg.norm(d, axis=-1)
    ii, jj = np.where((r < 2.7) & (r > 1e-6))
    g = nx.Graph(); g.add_nodes_from(range(len(pos)))
    for i, j in zip(ii, jj):
        if i < j: g.add_edge(int(i), int(j))
    R = sp_rings(g, 7)
    counts = Counter(len(x) for x in R)
    assert counts.get(6, 0) == 2*len(pos), (counts.get(6, 0), 2*len(pos))
    assert set(counts) == {6}, dict(counts)


@check("cycle_basis is shown to disagree with the truth, as documented")
def test_cycle_basis_is_shown_to_disagree_with_the_truth_as_documen():
    import networkx as nx
    from icemotifs.rings import sp_rings
    from collections import Counter
    a = 5.43
    b = np.array([[0,0,0],[0,.5,.5],[.5,0,.5],[.5,.5,0],
                  [.25,.25,.25],[.25,.75,.75],[.75,.25,.75],[.75,.75,.25]])
    c = np.array([[i,j,k] for i in range(3) for j in range(3) for k in range(3)])
    pos = ((c[:,None,:]+b[None,:,:]).reshape(-1,3))*a
    L = 3*a
    d = pos[:,None,:]-pos[None,:,:]; d -= L*np.rint(d/L)
    r = np.linalg.norm(d, axis=-1)
    ii, jj = np.where((r < 2.7) & (r > 1e-6))
    g = nx.Graph(); g.add_nodes_from(range(len(pos)))
    for i, j in zip(ii, jj):
        if i < j: g.add_edge(int(i), int(j))
    truth = len(sp_rings(g, 7))
    basis6 = Counter(len(x) for x in nx.cycle_basis(g)).get(6, 0)
    # this is the finding, asserted so it cannot quietly change
    assert basis6 < 0.25 * truth, (basis6, truth)


@check("published split reproduces Table S1's three counts exactly")
def test_published_split_reproduces_table_s1_s_three_counts_exactly():
    from icemotifs import published as P
    tr, va, te = P.split_published(18058040)
    assert len(tr) == 2407738, len(tr)
    assert len(va) == 1203870, len(va)
    assert len(te) == 14446432, len(te)
    assert len(tr) + len(va) + len(te) == 18058040
    # and it puts exactly 1338 of the 10,040 reference rows into training
    assert int((tr < 10040).sum()) == 1338, int((tr < 10040).sum())


@check("published split is NOT frame-disjoint; split_by_block is")
def test_published_split_is_not_frame_disjoint_split_by_block_is():
    from icemotifs import published as P
    n = 141 * 1000
    tr, va, te = P.split_published(n)
    blk = lambda idx: set((idx // 1000).tolist())
    # the stride split touches every block in all three sets
    assert len(blk(tr) & blk(te)) == 141, len(blk(tr) & blk(te))
    tr2, va2, te2 = P.split_by_block(n, n_blocks=141)
    assert not (blk(tr2) & blk(te2)), "block split leaked a block"
    assert not (blk(va2) & blk(te2))


@check("fit_scalers masks sentinel targets; the original did not")
def test_fit_scalers_masks_sentinel_targets_the_original_did_not():
    from icemotifs import published as P
    from icemotifs import columns as C
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (5000, 281)).astype(np.float32)
    X[:, 279] = rng.normal(32.0, 2.2, 5000)
    X[:, 280] = rng.normal(-0.417, 0.0295, 5000)
    X[:50, 280] = 1000.0                       # 1% sentinels
    masked = P.fit_scalers(X, C.LAYOUT_281, sentinel=1000.0, verbose=False)
    raw = P.fit_scalers(X, C.LAYOUT_281, sentinel=None, verbose=False)
    sd_masked = masked["targets"][1][1]
    sd_raw = raw["targets"][1][1]
    assert 0.02 < sd_masked < 0.04, sd_masked          # the real spread
    assert sd_raw > 50, sd_raw                          # blown up by sentinels
    print(f"      energy sd: masked {sd_masked:.4f}, unmasked {sd_raw:.1f}")


@check("cyclical_beta ramps then holds, with the published period")
def test_cyclical_beta_ramps_then_holds_with_the_published_period():
    from icemotifs import published as P
    b = [P.cyclical_beta(i) for i in range(20000)]
    assert b[0] == 0.0
    assert abs(b[5000] - 1e-3) < 1e-9, b[5000]
    assert abs(b[7500] - 1e-3) < 1e-9                   # held over the second half
    assert abs(b[9999] - 1e-3) < 1e-9
    assert b[10001] < 1e-4                              # and resets
    assert P.cyclical_beta(12345, interval=0) == 1e-3   # constant mode



# ------------------------------------------------------- added 10 Sep

@check("a frame window reads exactly its own frames, index or not")
def test_a_frame_window_reads_exactly_its_own_frames_index_or_not():
    from ase.build import bulk

    a = bulk("Si", "diamond", a=5.431, cubic=True).repeat((2, 2, 2))
    path = os.path.join(TMP, "seek.dump")
    with open(path, "w") as fh:
        for i in range(5):
            c = np.array(a.get_cell())
            fh.write(f"ITEM: TIMESTEP\n{i}\nITEM: NUMBER OF ATOMS\n{len(a)}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            for k in range(3):
                fh.write(f"0.0 {c[k][k]:.6f}\n")
            fh.write("ITEM: ATOMS id type x y z c_pe\n")
            for j, p in enumerate(a.positions):
                fh.write(f"{j+1} 1 {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} -4.3\n")

    off = lammps.frame_offsets(path, cache=os.path.join(TMP, "seek_idx.json"))
    assert len(off) == 5, off
    sd = S.get("silicon-sw")
    seek = [2 + i for i, _ in enumerate(
        lammps.iter_dump(path, sd, start_frame=2, offsets=off))]
    assert seek == [2, 3, 4], seek
    fall = [2 + i for i, _ in enumerate(
        lammps.iter_dump(path, sd, start_frame=2))]
    assert fall == [2, 3, 4], fall          # fallback must agree with the seek


@check("triangles appear only when the ring cutoff crosses the 2nd shell")
def test_triangles_appear_only_when_the_ring_cutoff_crosses_the_2nd():
    from ase.build import bulk
    from icemotifs import rings as R

    d = bulk("Si", "diamond", a=5.431, cubic=True).repeat((3, 3, 3))
    rows = {r["r_cut"]: r for r in R.cutoff_scan(d, [2.72, 3.30, 3.90])}
    # first shell: the analytic answer, 432 six-rings and nothing shorter
    assert rows[2.72]["triangles"] == 0, rows[2.72]
    assert rows[2.72]["R6"] == 432, rows[2.72]
    assert rows[3.30]["triangles"] == 0, rows[3.30]     # wide safe plateau
    # second shell: 3,024 triangles out of nowhere, six-rings gone. This is the
    # fingerprint of POSCAR.Ic.opt's 1,109 trimers in the released ring table.
    assert rows[3.90]["triangles"] > 0, rows[3.90]
    assert rows[3.90]["R6"] == 0, rows[3.90]


@check("100k blocks load without running out of file descriptors")
def test_100k_blocks_load_without_running_out_of_file_descriptors():
    import resource

    from icemotifs import data as D

    d = os.path.join(TMP, "manyblocks")
    os.makedirs(d, exist_ok=True)
    for i in range(900):
        np.save(os.path.join(d, "frame_%06d.npy" % i),
                np.zeros((4, C.LAYOUT_281.n_cols), np.float32))

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    try:
        # The pattern that failed: one open memory-map per block, all held at
        # once. 900 blocks against 256 descriptors reproduces the Gartner
        # stage-2 failure (100,125 blocks) in two seconds.
        try:
            _ = [np.load(os.path.join(d, f), mmap_mode="r")
                 for f in sorted(os.listdir(d))]
            raise AssertionError("the fd limit was not tight enough to test")
        except OSError as exc:
            assert exc.errno == 24, exc          # ENFILE/EMFILE: too many open files

        X, layout = D.load_features(d, verbose=False)
        assert X.shape == (3600, C.LAYOUT_281.n_cols), X.shape
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))



@check("--beta-max and --lambda-pred are independent, and default to published")
def test_beta_max_and_lambda_pred_are_independent_and_default_to_pu():
    # The original code computed lambda3 = 1/BETA from a single --BETA knob, so
    # "beta" is ambiguous in the literature around this model. Here they are two
    # separate weights and must stay that way: sweeping one must not move the
    # other. A regression here would silently confound a KLD sweep with a
    # property-head sweep.
    from icemotifs.pipeline import train_vae as T

    pub = T.PUBLISHED
    assert pub["beta_max"] == 1e-3, pub["beta_max"]
    assert pub["lambda3"] == 0.1, pub["lambda3"]

    ap = T._parser() if hasattr(T, "_parser") else None
    if ap is None:                      # parser is built inside main()
        import argparse, inspect
        src = inspect.getsource(T)
        assert '"--beta-max"' in src, "--beta-max is not exposed"
        assert '"--lambda-pred"' in src, "--lambda-pred is not exposed"
        # and each must write only its own cfg key
        assert 'cfg["beta"] = a.beta_max' in src
        assert 'cfg["lambda3"] = a.lambda_pred' in src
        assert 'cfg["lambda3"] = a.beta_max' not in src
        assert 'cfg["beta"] = a.lambda_pred' not in src

@check("a dead target column stops training, unless the head is off")
def test_a_dead_target_column_stops_training_unless_the_head_is_off():
    from icemotifs import published as P, scalers as SC

    lay = C.LAYOUT_281
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (800, lay.n_cols))
    X[:, 279] = rng.normal(29.9, 1.17, 800)      # voronoi: real data
    X[:, 280] = np.nan                            # energy: dead everywhere

    # property head ON -- this guard is what catches a NaN energy column
    try:
        P.fit_scalers(X, lay, verbose=False)
        raise AssertionError("an all-NaN target must stop the run")
    except SystemExit:
        pass

    # property head OFF -- the guard must not refuse its own remedy
    out = P.fit_scalers(X, lay, verbose=False, allow_dead_targets=True)
    m, sd = out["targets"]
    assert np.isfinite(m).all() and np.isfinite(sd).all(), (m, sd)
    assert abs(m[0] - 29.9) < 0.3 and abs(sd[0] - 1.17) < 0.3, m
    assert (m[1], sd[1]) == (0.0, 1.0), "dead column not neutralised"

    # and scalers.fit must never write NaN into a .npz
    A = rng.normal(3, 2, (400, 5))
    clean = SC.fit(A)
    assert np.allclose(clean.mean.numpy(), A.mean(0))
    assert np.allclose(clean.std.numpy(), A.std(0))
    A[:, 2] = np.nan
    dirty = SC.fit(A)
    assert np.isfinite(dirty.mean.numpy()).all()
    assert np.isfinite(dirty.std.numpy()).all()


@check("the published checkpoint accepts water cutoffs and refuses Si's")
def test_the_published_checkpoint_accepts_water_cutoffs_and_refuses():
    from icemotifs import published as P

    # The two water systems reproduce the published 6.0 A exactly; that is the
    # whole reason the published weights can read their features at all.
    for name in ("water-mw", "water-mlbop"):
        ok, _lines = P.cutoff_report(S.get(name))
        assert ok, name
    # Si is the same reduced cutoffs on a different d_nn, so a different
    # radius per column. Encoding it with the published weights is a transfer
    # experiment and has to be asked for explicitly.
    ok, lines = P.cutoff_report(S.get("silicon-sw"))
    assert not ok
    assert any("DIFFERENT input space" in ln for ln in lines), lines


@check("domain_report separates in-distribution rows from extrapolation")
def test_domain_report_separates_in_distribution_rows_from_extrapol():
    from icemotifs import published as P, scalers as SC

    lay = C.LAYOUT_281
    rng = np.random.default_rng(7)
    mean_st, sd_st = rng.normal(0, 1, 27), rng.uniform(0.5, 2, 27)
    mean_so, sd_so = rng.normal(0, 1, 252), rng.uniform(0.5, 2, 252)
    scal = (SC.TorchStandardScaler(mean_st, sd_st),
            SC.TorchStandardScaler(mean_so, sd_so), None)

    X = np.empty((4000, 281), np.float32)
    X[:, lay.stein] = rng.normal(mean_st, sd_st, (4000, 27))
    X[:, lay.soap] = rng.normal(mean_so, sd_so, (4000, 252))
    near, _ = P.domain_report(X, lay, scal)
    assert near < 2, near                       # drawn from the same law

    X[:, :279] += 40.0                          # shove it far outside
    far, lines = P.domain_report(X, lay, scal)
    assert far > 5, far
    assert any("OUTSIDE" in ln for ln in lines), lines


@check("want_recon=False skips the (n,281) array without changing Z")
def test_want_recon_false_skips_the_n_281_array_without_changing_z():
    import torch

    from icemotifs import encode as E, model as M, scalers as SC

    lay = C.LAYOUT_281
    rng = np.random.default_rng(11)
    st = SC.TorchStandardScaler(np.zeros(27), np.ones(27))
    so = SC.TorchStandardScaler(np.zeros(252), np.ones(252))
    tg = SC.TorchStandardScaler(np.zeros(2), np.ones(2))
    torch.manual_seed(0)
    model = M.build(st, so, dim_z=40, regression=2)
    X = rng.normal(0, 1, (300, 281)).astype(np.float32)

    Z1, MU1, LV1, RE1, m1 = E.encode_array(model, X, lay, (st, so, tg),
                                           batch_size=128, want_recon=True)
    Z2, MU2, LV2, RE2, m2 = E.encode_array(model, X, lay, (st, so, tg),
                                           batch_size=128, want_recon=False)
    assert RE1.shape == (300, 281)
    assert RE2 is None
    assert np.array_equal(Z1, Z2) and np.array_equal(MU1, MU2)
    assert np.array_equal(LV1, LV2)
    # the per-target split is what makes the published regrsca readable
    assert set(m1["mse_per_target"]) == {"voronoi_volume", "energy"}
    assert m1["mse_per_target"] == m2["mse_per_target"]


if __name__ == "__main__":
    passed, failed = 0, []
    for fn in CHECKS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.check_name}")
        except Exception as exc:                                    # noqa: BLE001
            failed.append((fn.check_name, exc))
            print(f"  FAIL  {fn.check_name}: {exc}")
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"{passed} passed, {len(failed)} failed")
    for name, exc in failed:
        print(f"  FAILED {name}: {exc}")
    sys.exit(1 if failed else 0)
