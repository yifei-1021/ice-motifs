#!/usr/bin/env python3
"""fix_drift.py -- remove centre-of-mass drift from an NPT ice/water trajectory.

WHERE THE DRIFT COMES FROM
--------------------------
`fix hold ice setforce 0.0 0.0 0.0` in stage 1 throws away the forces on the
template. The water pushes on the ice, the ice's reaction is discarded, and over
100,000 steps the water group accumulates a net momentum. At the start of
production `velocity ice create ... ` re-zeroes the ICE group's momentum
(velocity create defaults to mom yes) but `velocity water scale` only rescales
the water's speeds -- it preserves the direction of its velocities, and with them
the accumulated drift. The cell therefore translates at roughly constant velocity
for the whole run. In the first campaign this reached ~350 A of displacement over
50 ns, which is what made `compute msd` unusable.

Prevention, one line in the deck, just before `reset_timestep 0`:

    velocity        all zero linear

This script repairs trajectories written before that line existed.

WHY FRACTIONAL COORDINATES
--------------------------
Under `npt tri` the cell changes volume AND shape, and the dump origin moves with
the tilt. Subtracting a Cartesian offset per frame therefore mixes the drift with
the affine box change. Instead every frame is converted to fractional coordinates
s = H^-1 (r - origin), unwrapped frame-to-frame in s (wrapping each displacement
into [-0.5, 0.5)), and the reference group's mean fractional displacement is
subtracted. That is immune to both the wrapping and the box deformation.

The frame-to-frame unwrap assumes no atom moves more than half a cell between
frames. At 50 ps spacing, 210 K and ~0.007 A/ps of drift that is ~0.4 A against a
~44 A half-cell, so there is a factor of 100 of headroom; the script checks it
anyway and warns.

USAGE
-----
    python3 fix_drift.py NPT.T210.s123456.dump -o NPT.T210.s123456.fixed.extxyz
    python3 fix_drift.py traj.extxyz --ref ice --mode unwrap --msd msd.dat

    --ref   all | ice | water   whose COM is held still          (default: all)
    --mode  wrap | unwrap       wrap back into the cell for OVITO, or leave the
                                trajectory continuous for MSD    (default: wrap)
    --out-format extxyz | dump                                   (default: extxyz)
    --msd FILE                  write drift-corrected MSD per group
    --affine current | initial  build Cartesian output from each frame's own cell,
                                or from frame 0's cell so the affine box change is
                                removed too                      (default: current)
    --every N                   keep only every Nth frame
    --max-frames N              stop after N kept frames

Reads and writes streaming, one frame at a time, so a 350 GB campaign is fine.
"""

import argparse
import re
import sys

import numpy as np

LATTICE_RE = re.compile(r'Lattice\s*=\s*"([^"]*)"')
PROPS_RE = re.compile(r'Properties\s*=\s*(\S+)')


# ----------------------------------------------------------------- cell helpers
def cell_from_dump_bounds(rows):
    """LAMMPS dump BOX BOUNDS -> (H, origin).

    Triclinic dumps print *bounding box* limits, not xlo/xhi, so the tilts have
    to be subtracted back out:
        xlo = xlo_bound - min(0, xy, xz, xy+xz)
        xhi = xhi_bound - max(0, xy, xz, xy+xz)
        ylo = ylo_bound - min(0, yz)
        yhi = yhi_bound - max(0, yz)
    H has the cell vectors as COLUMNS, so r = origin + H @ s.
    """
    if len(rows[0]) == 3:
        (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo_b, zhi_b, yz) = rows
        xlo = xlo_b - min(0.0, xy, xz, xy + xz)
        xhi = xhi_b - max(0.0, xy, xz, xy + xz)
        ylo = ylo_b - min(0.0, yz)
        yhi = yhi_b - max(0.0, yz)
        zlo, zhi = zlo_b, zhi_b
    else:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = rows
        xy = xz = yz = 0.0
    a = np.array([xhi - xlo, 0.0, 0.0])
    b = np.array([xy, yhi - ylo, 0.0])
    c = np.array([xz, yz, zhi - zlo])
    return np.column_stack([a, b, c]), np.array([xlo, ylo, zlo])


def ice_ids_from_data(path):
    """Reproduce the deck's `region r_ice` / `group ice` EXACTLY.

    The deck assigns the group from the coordinates in <NAME>.init.data, before
    the minimisation and stage 1. Selecting the same z band in frame 0 of the
    dump instead gives a slightly different set, because atoms have since moved
    across the band edges -- 7050 vs the deck's 6800 for exp.12_1_11. Anything
    compared against c_f4w / c_q6w has to use the deck's set, so read it from the
    data file.
    """
    zlo = zhi = None
    ids, zs = [], []
    with open(path) as fh:
        lines = fh.readlines()
    for ln in lines:
        if ln.rstrip().endswith("zlo zhi"):
            zlo, zhi = (float(v) for v in ln.split()[:2])
            break
    if zlo is None:
        sys.exit(f"{path}: no 'zlo zhi' line found")
    i = next((k for k, ln in enumerate(lines) if ln.strip().startswith("Atoms")), None)
    if i is None:
        sys.exit(f"{path}: no Atoms section")
    for ln in lines[i + 1:]:
        f = ln.split()
        if not f:
            continue
        if not f[0].lstrip("-").isdigit():
            break
        ids.append(int(f[0]))
        zs.append(float(f[4]))          # atom_style atomic: id type x y z
    ids = np.array(ids, np.int64)
    zs = np.array(zs)
    Lz = zhi - zlo
    sel = (zs >= zlo + Lz / 3.0) & (zs <= zlo + 2.0 * Lz / 3.0)
    return set(ids[sel].tolist())


# -------------------------------------------------------------------- readers
def read_dump(path):
    """Yield (step, H, origin, ids, types, pos, extra, extra_names)."""
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()                      # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            fh.readline()                      # ITEM: BOX BOUNDS ...
            rows = [[float(v) for v in fh.readline().split()] for _ in range(3)]
            H, origin = cell_from_dump_bounds(rows)
            cols = fh.readline().split()[2:]   # ITEM: ATOMS id type x y z ...
            data = np.empty((n, len(cols)))
            truncated = False
            for i in range(n):
                vals = np.fromstring(fh.readline(), sep=" ")
                if vals.size != len(cols):
                    # A job still running leaves its last frame half-written.
                    # Stop cleanly instead of crashing on it.
                    print(f"  note: frame at step {step} is incomplete "
                          f"({i}/{n} atoms) -- trajectory still being written, "
                          f"stopping here", file=sys.stderr)
                    truncated = True
                    break
                data[i] = vals
            if truncated:
                return
            ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
            ids = data[:, cols.index("id")].astype(np.int64)
            types = data[:, cols.index("type")].astype(np.int32)
            keep = [k for k in range(len(cols))
                    if cols[k] not in ("id", "type", "x", "y", "z",
                                       "xs", "ys", "zs", "xu", "yu", "zu")]
            yield (step, H, origin, ids, types,
                   data[:, [ix, iy, iz]], data[:, keep], [cols[k] for k in keep])


def read_extxyz(path):
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head.strip():
                return
            n = int(head)
            comment = fh.readline()
            m = LATTICE_RE.search(comment)
            if not m:
                sys.exit("extxyz frame has no Lattice=\"...\" in its comment line")
            L = np.array([float(v) for v in m.group(1).split()]).reshape(3, 3)
            H = L.T                            # Lattice is row-major a,b,c
            origin = np.zeros(3)
            step = 0
            ms = re.search(r'(?:Time|Timestep|step)\s*=\s*([-\d.eE+]+)', comment)
            if ms:
                step = float(ms.group(1))

            names, kinds = [], []
            mp = PROPS_RE.search(comment)
            if mp:
                f = mp.group(1).split(":")
                for k in range(0, len(f) - 2, 3):
                    names.append(f[k])
                    kinds.append((f[k + 1], int(f[k + 2])))
            else:
                names, kinds = ["species", "pos"], [("S", 1), ("R", 3)]

            raw = [fh.readline().split() for _ in range(n)]
            if any(len(r) < 4 for r in raw):
                print(f"  note: last extxyz frame is incomplete -- stopping here",
                      file=sys.stderr)
                return
            col, extra, extra_names = 0, [], []
            pos = None
            for name, (kind, w) in zip(names, kinds):
                block = [r[col:col + w] for r in raw]
                if name == "pos":
                    pos = np.array(block, float)
                elif kind != "S":
                    for j in range(w):
                        extra.append(np.array([b[j] for b in block], float))
                        extra_names.append(name if w == 1 else f"{name}{j}")
                col += w
            ids = np.arange(1, n + 1, dtype=np.int64)
            types = np.ones(n, np.int32)
            ex = np.column_stack(extra) if extra else np.empty((n, 0))
            yield step, H, origin, ids, types, pos, ex, extra_names


# -------------------------------------------------------------------- writers
def safe_name(nm):
    """c_q6[1] -> c_q6_1 : brackets and dots break extxyz Properties parsing."""
    return re.sub(r"[^A-Za-z0-9_]", "_", nm).strip("_")


def write_extxyz(fh, step, H, pos, extra, extra_names, is_ice, species="O"):
    lat = " ".join(f"{v:.8f}" for v in H.T.reshape(-1))
    props = "species:S:1:pos:R:3:ice:I:1"
    for nm in extra_names:
        props += f":{safe_name(nm)}:R:1"
    fh.write(f"{len(pos)}\n")
    fh.write(f'Lattice="{lat}" Properties={props} Timestep={step} '
             f'pbc="T T T"\n')
    for i in range(len(pos)):
        row = f"{species} {pos[i,0]:.5f} {pos[i,1]:.5f} {pos[i,2]:.5f} {int(is_ice[i])}"
        if extra.shape[1]:
            row += " " + " ".join(f"{v:.6g}" for v in extra[i])
        fh.write(row + "\n")


def write_dump(fh, step, H, origin, ids, types, pos, extra, extra_names):
    a, b, c = H[:, 0], H[:, 1], H[:, 2]
    xy, xz, yz = b[0], c[0], c[1]
    xlo, ylo, zlo = origin
    xhi, yhi, zhi = xlo + a[0], ylo + b[1], zlo + c[2]
    fh.write("ITEM: TIMESTEP\n%d\n" % step)
    fh.write("ITEM: NUMBER OF ATOMS\n%d\n" % len(pos))
    fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    fh.write(f"{xlo + min(0.0, xy, xz, xy+xz):.10e} "
             f"{xhi + max(0.0, xy, xz, xy+xz):.10e} {xy:.10e}\n")
    fh.write(f"{ylo + min(0.0, yz):.10e} {yhi + max(0.0, yz):.10e} {xz:.10e}\n")
    fh.write(f"{zlo:.10e} {zhi:.10e} {yz:.10e}\n")
    fh.write("ITEM: ATOMS id type x y z " + " ".join(extra_names) + "\n")
    for i in range(len(pos)):
        row = (f"{ids[i]} {types[i]} "
               f"{pos[i,0]:.5f} {pos[i,1]:.5f} {pos[i,2]:.5f}")
        if extra.shape[1]:
            row += " " + " ".join(f"{v:.6g}" for v in extra[i])
        fh.write(row + "\n")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ref", choices=["all", "ice", "water"], default="all")
    ap.add_argument("--ice-from-data", metavar="FILE",
                    help="<NAME>.init.data -- take the template atom IDs from the "
                         "data file, reproducing the deck's `group ice` exactly")
    ap.add_argument("--mode", choices=["wrap", "unwrap"], default="wrap")
    ap.add_argument("--out-format", choices=["extxyz", "dump"], default="extxyz")
    ap.add_argument("--affine", choices=["current", "initial"], default="current")
    ap.add_argument("--msd")
    ap.add_argument("--dt", type=float, default=0.005,
                    help="ps per timestep, for the MSD time column (default 0.005)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    reader = read_extxyz if args.traj.endswith((".extxyz", ".xyz")) else read_dump

    H0 = order = is_ice = s_prev = s_unwrap = s_ref0 = s_com0 = None
    out = open(args.out, "w")
    msd_fh = open(args.msd, "w") if args.msd else None
    if msd_fh:
        msd_fh.write("# t(ps) step msd_all msd_water msd_ice "
                     "drift_x drift_y drift_z |drift|   (A^2, A)\n")

    kept = nread = 0
    maxjump = 0.0
    for step, H, origin, ids, types, pos, extra, extra_names in reader(args.traj):
        nread += 1
        if (nread - 1) % args.every:
            continue

        if order is None:
            order = np.argsort(ids)
            H0, origin0 = H.copy(), origin.copy()
            # Reproduce the deck's grouping EXACTLY: middle third in z of the
            # first frame's cell is the template.
            if args.ice_from_data:
                ice_set = ice_ids_from_data(args.ice_from_data)
                is_ice = np.isin(ids[order], np.fromiter(ice_set, np.int64))
                src = args.ice_from_data
            else:
                z = pos[order][:, 2]
                Lz = H[2, 2]
                z1, z2 = origin[2] + Lz / 3.0, origin[2] + 2.0 * Lz / 3.0
                is_ice = (z >= z1) & (z <= z2)
                src = "frame-0 z band"
                if not args.quiet:
                    print("  note: template taken from the frame-0 z band, which "
                          "is NOT the deck's `group ice` (atoms have moved across "
                          "the edges during stage 1). Pass --ice-from-data "
                          "<NAME>.init.data to match c_f4w / c_q6w exactly.",
                          file=sys.stderr)
            if not args.quiet:
                print(f"frame 0: {is_ice.sum()} template atoms, "
                      f"{(~is_ice).sum()} liquid atoms  [{src}]", file=sys.stderr)
        elif not np.array_equal(ids[order], ids[np.argsort(ids)]):
            order = np.argsort(ids)

        pos = pos[order]
        extra = extra[order]
        s = (np.linalg.inv(H) @ (pos - origin).T).T

        if s_prev is None:
            s_unwrap = s.copy()
        else:
            ds = s - s_prev
            ds -= np.round(ds)                 # minimum image, in fractional space
            # Check AFTER wrapping: a raw ds near +-1 is just an atom crossing a
            # boundary and is exactly what the wrap is for. Only a large residual
            # means the frames really are too far apart to unwrap unambiguously.
            jump = np.abs(ds).max()
            maxjump = max(maxjump, jump)
            if jump > 0.4 and not args.quiet:
                print(f"  warning: step {step} has a wrapped fractional "
                      f"displacement of {jump:.3f}; frames may be too far apart "
                      f"to unwrap reliably", file=sys.stderr)
            s_unwrap = s_unwrap + ds
        s_prev = s

        sel = {"all": slice(None), "ice": is_ice, "water": ~is_ice}[args.ref]
        # +0.5 puts the reference group's COM at the CENTRE of the cell rather
        # than at fractional 0, so a wrapped slab renders whole in the middle of
        # the box instead of split across the z boundary. A constant offset, so
        # it cancels out of every displacement and leaves the MSD untouched.
        s_fixed = s_unwrap - s_unwrap[sel].mean(axis=0) + 0.5

        if s_ref0 is None:
            s_ref0 = s_fixed.copy()

        # extxyz has no origin field -- its Lattice implies an origin at 0 -- and
        # the LAMMPS dump origin MOVES with the tilt (xlo_bound goes negative), so
        # carrying it through would reinject a per-frame offset. Build positions
        # origin-free and only add an origin back for dump output.
        Hout = H if args.affine == "current" else H0
        origin_out = origin if args.affine == "current" else origin0
        s_out = s_fixed - np.floor(s_fixed) if args.mode == "wrap" else s_fixed
        r = (Hout @ s_out.T).T

        if msd_fh:
            # MSD from the drift-corrected, unwrapped trajectory, relative to the
            # first kept frame. s_fixed already has the reference COM removed, so
            # this is pure internal motion.
            sq = ((Hout @ (s_fixed - s_ref0).T).T ** 2).sum(axis=1)
            # and, for the record, how much drift was taken out: the reference
            # group's own COM displacement since the first kept frame.
            s_com = s_unwrap[sel].mean(axis=0)
            if s_com0 is None:
                s_com0 = s_com.copy()
            dr = Hout @ (s_com - s_com0)
            msd_fh.write(f"{step*args.dt:12.3f} {step:10d} "
                         f"{sq.mean():14.5f} {sq[~is_ice].mean():14.5f} "
                         f"{sq[is_ice].mean():14.5f} "
                         f"{dr[0]:10.4f} {dr[1]:10.4f} {dr[2]:10.4f} "
                         f"{np.linalg.norm(dr):10.4f}\n")

        if args.out_format == "extxyz":
            write_extxyz(out, step, Hout, r, extra, extra_names, is_ice)
        else:
            write_dump(out, step, Hout, origin_out, ids[order], types[order],
                       r + origin_out, extra, extra_names)

        kept += 1
        if not args.quiet and kept % 50 == 0:
            print(f"  {kept} frames written (step {step})", file=sys.stderr)
        if args.max_frames and kept >= args.max_frames:
            break

    out.close()
    if msd_fh:
        msd_fh.close()
    if not args.quiet:
        print(f"wrote {kept} frames to {args.out} "
              f"(ref={args.ref}, mode={args.mode}, affine={args.affine}); "
              f"largest pre-wrap fractional step {maxjump:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()