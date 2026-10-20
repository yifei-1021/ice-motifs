#!/usr/bin/env python3
"""
STAGE 1 -- featurise one trajectory, in whatever format it arrives in.

Supersedes `featurize_lammps.py`, which only read LAMMPS dumps. The three
formats in the re-analysis set are all handled here so that every dataset goes
through ONE code path and therefore one set of cutoff decisions:

    --kind dump       LAMMPS text dump          heterogenous/, mW-nucleation/, Si-nucleation/
    --kind extxyz     one multi-frame extxyz    Gartner-LDL-HDL/
    --kind filelist   `ls -R` listing of        mW-nucleation-N64-0.2K/
                      per-frame extxyz files

Output, per selected frame, one .npy of shape (n_atoms, 281):

    [  0: 27)  averaged Steinhardt (Lechner-Dellago), 3 cutoffs x 9 l
    [ 27:279)  SOAP, n_max = 8, l_max = 6, sigma = 1.0 A, gto, periodic
    [279:280)  Voronoi cell volume
    [280:281)  per-atom potential energy

the same layout as the published water run, so stages 2-5 are unchanged.

FRAME WINDOWS
-------------
`--frame-start/--frame-end` featurise a half-open window of the trajectory, and
outputs are named by ABSOLUTE frame index, so N windows of the same trajectory
can run concurrently and pool into one directory with no collisions. This is the
same pattern as run_frame_windows.sh, and it is what makes the 64,000-atom
frames tractable.

d_nn -- THE ONE THING THAT MUST NOT BE GUESSED
----------------------------------------------
Every cutoff is a fixed multiple of the first-neighbour distance (see
icemotifs/systems.py), so that water reproduces the published 3.2/4.8/6.0 A and
every other system is measured at the SAME coordination shells rather than the
same absolute radius. Featurising silicon at the water cutoffs would put a
different number of shells inside r_cut and any difference found would be an
artefact of the cutoff.

This script therefore MEASURES d_nn from the first frame of the window and
records it in the metadata. If it disagrees with the registry default by more
than --dnn-tol (3% by default) it stops, because that is the signature of either
the wrong --system or a state point whose structure has genuinely shifted (which
is exactly the case for the ab initio HDL/LDL water). Override deliberately with
--d-nn, and the measured value is still recorded next to it.

SCALING
-------
Nothing is standard-scaled here. The scaler is fitted on the training split only
and reused at inference; stage 2 does that. Writing pre-scaled features is how a
scaler ends up fitted on the test set.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
import time

import numpy as np

# NOTE: this module used to walk up the directory tree looking for a sibling
# `icemotifs/` and prepend it to sys.path, so it could run from inside the
# source tree. It is a package module now; the plain import below resolves
# through the installed distribution, and the old search would have let an
# unrelated checkout two directories up shadow it.

from icemotifs import columns as C           # noqa: E402
from icemotifs import descriptors as D       # noqa: E402
from icemotifs import lammps, systems as S   # noqa: E402


# ------------------------------------------------------------------ diagnostics
#
# Windows die inside `xargs -P`, in their own log, with only "FAIL" reaching the
# job log. Whatever a window has to say, it has to say in writing and before it
# dies -- so everything below exists to make a dead window explain itself.

def _rss_mb():
    """Resident set size of THIS process, MB. /proc first (exact, no deps)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return float("nan")


def _reclaim():
    """Give the memory of the frame just finished back to the OS.

    Two separate things hold it otherwise, and both showed up in the Si logs as
    RSS climbing 515 -> 1778 -> 2276 -> 2652 MB over three frames, until the
    window was killed with no traceback (SIGKILL leaves nothing to catch):

      * REFERENCE CYCLES. pyscal3's System and its neighbour data form cycles,
        so they are freed by the cycle collector rather than by refcount, i.e.
        not at the end of the frame. gc.collect() finds ~150 objects per frame.
      * GLIBC ARENAS. Even once freed, the large blocks numpy took stay in the
        process's arenas rather than returning to the OS, so RSS never falls.
        malloc_trim(0) hands them back.

    Together they hold RSS flat instead of growing ~450 MB per frame -- which is
    the difference between 10 windows fitting in a 40 GB job and not.
    """
    import gc
    gc.collect()
    try:                                      # glibc only; a no-op elsewhere
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def _mem_limit_mb():
    """The cgroup memory ceiling this process actually lives under, if any.

    PBS enforces `mem=40GB` through cgroups, and being killed by that ceiling is
    indistinguishable from any other SIGKILL unless you know what the ceiling
    was. cgroup v2 first, then v1.
    """
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as fh:
                v = fh.read().strip()
            if v and v != "max":
                v = int(v)
                if v < (1 << 60):
                    return v / 2**20
        except Exception:
            continue
    return None


def _install_crash_handlers():
    """Make a killed or crashed window leave a traceback behind.

    'Terminated' with nothing else in the log is the shell reporting SIGTERM.
    Python's default handler for it prints nothing at all, so there is no way
    to tell a time-limit kill from an OOM kill from a manual cancellation.
    faulthandler turns every one of those into a stack trace at the line it was
    executing -- which matters most when the work is split across parallel
    windows and only the word FAIL reaches the top-level log.
    """
    import faulthandler
    import signal
    faulthandler.enable()                     # SIGSEGV, SIGFPE, SIGABRT, SIGBUS
    for sig in ("SIGTERM", "SIGXCPU", "SIGUSR1", "SIGINT"):
        s = getattr(signal, sig, None)
        if s is None:
            continue
        try:
            faulthandler.register(s, chain=True)
        except Exception:
            pass

    def _on_term(signum, _frame):
        name = signal.Signals(signum).name
        why = {"SIGTERM": "scheduler time limit, cancellation, or the OOM killer",
               "SIGXCPU": "CPU-time limit",
               "SIGINT": "interrupt"}.get(name, "")
        print(f"\n!! killed by {name}" + (f" ({why})" if why else "") +
              f" -- RSS {_rss_mb():.0f} MB, peak {_peak_rss_mb():.0f} MB",
              flush=True)
        print("   it was here:", flush=True)
        try:
            # signal.signal() replaced faulthandler's own handler, so ask for
            # the stack explicitly -- the line it died on is the whole point.
            faulthandler.dump_traceback(file=sys.stdout)
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(143 if name == "SIGTERM" else 1)

    for sig in ("SIGTERM", "SIGXCPU"):
        s = getattr(signal, sig, None)
        if s is not None:
            try:
                signal.signal(s, _on_term)
            except Exception:
                pass


def _print_environment(a):
    """One block, at the top of every window log, of everything that differs
    between a window that works and a window that does not."""
    import platform
    import socket
    print("---- environment " + "-" * 55)
    print(f"  host {socket.gethostname()}   pid {os.getpid()}   "
          f"cwd {os.getcwd()}")
    jid = os.environ.get("PBS_JOBID")
    if jid:
        print(f"  PBS_JOBID {jid}   PBS_NODEFILE "
              f"{os.environ.get('PBS_NODEFILE', '-')}")
    print(f"  python {platform.python_version()} at {sys.executable}")
    vers = []
    for mod in ("numpy", "scipy", "ase", "pyscal3", "dscribe", "networkx",
                "sklearn", "torch"):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v is None:                     # pyscal3 and dscribe expose none
                try:
                    from importlib.metadata import version as _v
                    v = _v({"sklearn": "scikit-learn"}.get(mod, mod))
                except Exception:
                    v = "?"
            vers.append(f"{mod} {v}")
        except Exception as exc:
            vers.append(f"{mod} MISSING({type(exc).__name__})")
    print("  " + ", ".join(vers))

    thr = {k: os.environ.get(k, "unset") for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")}
    print("  threads: " + "  ".join(f"{k.split('_')[0]}={v}"
                                    for k, v in thr.items()))
    if all(v == "unset" for v in thr.values()):
        print("  !! no thread caps set. With N windows under `xargs -P N`, each"
              " library\n     may open N threads apiece -- N^2 threads and N"
              " times the scratch\n     memory. Set OMP_NUM_THREADS=1 for the"
              " windows.")
    try:
        print(f"  cpus visible {os.cpu_count()}   "
              f"affinity {len(os.sched_getaffinity(0))}")
    except Exception:
        print(f"  cpus visible {os.cpu_count()}")

    lim = _mem_limit_mb()
    free = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    free = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    print(f"  memory: RSS now {_rss_mb():.0f} MB"
          + (f"   cgroup limit {lim:.0f} MB" if lim else "   cgroup limit none")
          + (f"   node available {free:.0f} MB" if free else ""))
    print(f"  argv: {' '.join(sys.argv[1:])}")
    print("-" * 72, flush=True)


# --------------------------------------------------------------------- readers

def _parse_filelist(path, root=None):
    """Read an `ls`/`ls -R` listing into absolute paths.

    Handles the exact shape of extxyz-file-locations.txt: directory headers that
    end in ':', bare filenames beneath them, blank separators. A listing of
    plain paths (one per line) also works.
    """
    base = root or os.path.dirname(os.path.abspath(path))
    cur, out = base, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.endswith(":"):                       # directory header
                d = s[:-1]
                cur = d if os.path.isabs(d) else os.path.normpath(
                    os.path.join(base, d))
                continue
            out.append(s if os.path.isabs(s) else os.path.join(cur, s))
    keep = [p for p in out if p.endswith((".extxyz", ".xyz"))]
    if len(keep) != len(out):
        print(f"  note: ignored {len(out) - len(keep)} non-extxyz entries "
              f"in the listing", flush=True)
    # frame order is the listing order; sort so windows are reproducible
    return sorted(keep)


def _iter_frames(kind, path, sysdef, *, root=None, allow_missing_energy=False,
                 start=0, offsets=None, energy_column=None, stop=None,
                 keep=None):
    """Yield (index, ase.Atoms, energy_or_None), starting at frame `start`.

    Skipping matters: a trajectory featurised in W windows would otherwise have
    window k re-read the k*window frames before it, which for 70 windows of a
    multi-GB dump is ~35x the necessary I/O. Dumps seek via the byte index;
    extxyz and per-file listings skip cheaply.

    `keep` is the set of frame indices the caller actually wants. It matters for
    --kind filelist, where every frame is its OWN file: without it the reader
    parsed all 100 files between one strided frame and the next and the caller
    threw 99 of them away. On the N64 set (stride 100, 64,000 atoms per file)
    that is 100x the I/O and parsing, and it is why the d_nn probe looked like
    it had hung -- it was reading 401 files to measure five.
    """
    if kind == "dump":
        for i, (atoms, energy) in enumerate(
                lammps.iter_dump(path, sysdef,
                                 allow_missing_energy=allow_missing_energy,
                                 start_frame=start, offsets=offsets,
                                 energy_column=energy_column,
                                 stop_frame=stop)):
            yield start + i, atoms, energy
    elif kind == "extxyz":
        # ase must parse a multi-frame extxyz sequentially, so frames before
        # `start` cannot be skipped without reading them -- but frames after
        # the window can simply not be read at all. Without the stop, window 0
        # of a 3,000-frame file parses all 3,000.
        from ase.io import iread
        for k, atoms in enumerate(iread(path, format="extxyz", index=":")):
            if stop is not None and k >= stop:
                return
            if k < start:
                continue
            yield k, atoms, None
    elif kind == "filelist":
        from ase.io import read
        for k, f in enumerate(_parse_filelist(path, root)):
            if k < start:
                continue                               # no file even opened
            if keep is not None and k not in keep:
                continue                               # nor for a strided-out
            if stop is not None and k >= stop:
                return
            yield k, read(f, format="extxyz", index=0), None
    else:                                             # pragma: no cover
        raise SystemExit(f"unknown --kind {kind!r}")


def _n_frames(kind, path, root=None):
    if kind == "dump":
        return lammps.count_frames(path)
    if kind == "filelist":
        return len(_parse_filelist(path, root))
    n = 0                                             # extxyz: count cheaply
    with open(path) as fh:
        for line in fh:
            if line.strip().isdigit():
                n += 1
    return n


# ----------------------------------------------------------------- featurising

_species_note = {}


def _restrict_species(atoms, energy, sysdef):
    """Keep only the atoms the descriptors are defined on.

    The ab initio water of Gartner et al. is ALL-ATOM: every frame has H as
    well as O. Every descriptor here is defined on the oxygen network -- SOAP
    is built for a single species, the Steinhardt cutoffs are multiples of the
    O-O first-neighbour distance, and the Voronoi volume is the one whose
    inverse is compared against coarse-grained water in Fig. 4A. Handing the
    hydrogens to SOAP raises "atomic numbers are not defined for this
    descriptor: {1}".

    Dropping them is not a workaround, it is what makes the comparison mean
    anything: the coarse-grained mW and ML-BOP trajectories have one site per
    molecule, so an oxygen-only description of the ab initio data is the like
    for like one. It is also a fact the SI has to state.
    """
    syms = atoms.get_chemical_symbols()
    keep_sym = sysdef.symbol
    if all(x == keep_sym for x in syms):
        return atoms, energy
    import numpy as _np
    mask = _np.array([x == keep_sym for x in syms], dtype=bool)
    if not mask.any():
        raise SystemExit(
            f"no {keep_sym} atoms in this frame; it holds "
            f"{sorted(set(syms))}. --system is wrong for this trajectory.")
    key = tuple(sorted(set(syms)))
    if key not in _species_note:
        _species_note[key] = True
        from collections import Counter
        c = Counter(syms)
        print(f"  frame holds {dict(c)}; keeping the {int(mask.sum()):,} "
              f"{keep_sym} atoms only. Every descriptor here is defined on the "
              f"{keep_sym} network, and the coarse-grained trajectories this is "
              f"compared with have one site per molecule. SAY THIS IN THE SI.",
              flush=True)
    out = atoms[mask]
    if energy is not None:
        energy = _np.asarray(energy).reshape(-1)[mask]
    return out, energy


def featurize_frame(atoms, energy, sysdef, timings=None,
                    soap_sigma=1.0, soap_rbf="gto",
                    energy_column=None):
    """One frame -> the 281-column block.

    Uses icemotifs.descriptors rather than reimplementing, because that module
    already handles the two pyscal3 traps: ghost atoms when the cell is small
    relative to the cutoff (which silently emits extra rows), and the Voronoi
    volume, which calculate.voronoi_vector() breaks on pyscal3 3.3.2 + NumPy 2.
    Both matter more here than in the published run: these cells are smaller.

    Pass a dict as `timings` to get per-descriptor seconds and the RSS after
    each one. Which of the three descriptors a window died in is the single most
    useful fact about a failed window, and a traceback only gives it when the
    process lives long enough to raise -- an OOM kill does not.
    """
    t = timings if timings is not None else {}
    atoms, energy = _restrict_species(atoms, energy, sysdef)

    def _step(name, fn):
        t0, m0 = time.time(), _rss_mb()
        r = fn()
        t[name] = (time.time() - t0, _rss_mb(), _rss_mb() - m0)
        return r

    stein = _step("steinhardt",
                  lambda: D.steinhardt(atoms, cutoffs=sysdef.stein_cutoffs))
    # sigma is a SWEEP PARAMETER, not a constant. It was hard-coded to 1.0 here
    # (the DScribe default the published run used) while descriptors.soap has
    # always taken it as an argument, so featurize.py could not vary the one
    # SOAP knob R2-6 asks about.
    sp = _step("soap", lambda: D.soap(
        atoms, r_cut=sysdef.soap_rcut, n_max=C.SOAP_N_MAX,
        l_max=C.SOAP_L_MAX, sigma=soap_sigma, rbf=soap_rbf,
        species=(sysdef.symbol,), periodic=True))
    vol = _step("voronoi", lambda: D.voronoi_volume(atoms))
    if energy is None:                                # extxyz carries its own
        energy = D._frame_energy(atoms, len(atoms),
                                 energy_column=energy_column)
    return np.hstack([stein, sp,
                      np.asarray(vol).reshape(-1, 1),
                      np.asarray(energy).reshape(-1, 1)])


def _offsets_for(a):
    if a.kind != "dump":
        return None
    return lammps.frame_offsets(
        a.input, cache=os.path.join(a.out, "_meta", "frame_index.json"))


def _measure_dnn(a, sysdef, idx):
    """Median first-neighbour distance over the first --dnn-frames kept frames.

    Split out of _resolve_dnn so a PINNED d_nn can still be checked against the
    trajectory it is being applied to. Pinning is a decision, and a decision
    deserves a number next to it in the log -- especially for Gartner, where
    2.8 A is a choice rather than a measurement and the SI has to say so.
    """
    want, vals = set(idx[:max(1, a.dnn_frames)]), []
    if not want:
        return vals
    for k, atoms, _ in _iter_frames(a.kind, a.input, sysdef, root=a.root,
                                    allow_missing_energy=True,
                                    start=min(want), keep=want,
                                    offsets=_offsets_for(a),
                                    stop=max(want) + 1):
        if k in want:
            try:
                # Same restriction the descriptors get. On an all-atom frame
                # the first g(r) peak is the O-H bond (0.985 A), not the O-O
                # first-neighbour distance, so measuring before the restriction
                # reports a bond length and calls it d_nn.
                probe, _ = _restrict_species(atoms, None, sysdef)
                vals.append(float(S.measure_d_nn(probe)))
                print(f"  d_nn probe frame {k}: {vals[-1]:.4f} A "
                      f"({len(probe)} {sysdef.symbol} of {len(atoms)} atoms, "
                      f"RSS {_rss_mb():.0f} MB)", flush=True)
            except Exception as exc:
                print(f"  d_nn not measurable on frame {k}: {exc}", flush=True)
            _reclaim()
        if k >= max(want):
            break
    return vals


def _resolve_dnn(a, sysdef, idx):
    """Return (system-with-the-right-d_nn, measured_value).

    Order of authority: --d-nn  >  the cache file  >  a fresh measurement
    median-ed over --dnn-frames frames of this window.
    """
    # --d-nn FIRST. It used to come after the cache, which meant a pinned value
    # was silently ignored on any trajectory that already had a d_nn.json from
    # an earlier run -- exactly the case when you pin the value *because* the
    # first run measured something you did not want.
    if a.d_nn:
        print(f"  d_nn {a.d_nn:.4f} A given on the command line "
              f"(registry {sysdef.d_nn} A)")
        if a.dnn_file and os.path.exists(a.dnn_file):
            try:
                with open(a.dnn_file) as fh:
                    old = json.load(fh).get("d_nn_used")
                if old is not None and abs(old - a.d_nn) > 1e-9:
                    print(f"  (overriding the cached {old:.4f} A in "
                          f"{a.dnn_file}; rewriting it)")
            except Exception:
                pass
        measured = None
        if getattr(a, "dnn_check", False):
            vals = _measure_dnn(a, sysdef, idx)
            if vals:
                measured = float(np.median(vals))
                off = abs(measured - a.d_nn) / a.d_nn
                verdict = ("consistent" if off <= a.dnn_tol
                           else "!! DISAGREES with the pin")
                print(f"  measured {measured:.4f} A here (median of "
                      f"{len(vals)} frame(s)) -- pinned {a.d_nn:.4f} A is "
                      f"{100*off:.1f}% away: {verdict}")
                if off > a.dnn_tol:
                    print(f"     The pin is still used -- pinning is a "
                          f"deliberate choice and this is only a check -- but "
                          f"a gap this size belongs in the SI, or the pin "
                          f"should be revisited.")
        out = _retune(sysdef, a.d_nn)
        _write_dnn(a, out, measured=measured, n=0, pinned=True)
        return out, measured

    if a.dnn_file and os.path.exists(a.dnn_file):
        with open(a.dnn_file) as fh:
            c = json.load(fh)
        print(f"  d_nn {c['d_nn_used']:.4f} A from {a.dnn_file} "
              f"(measured once for this trajectory)")
        use, measured = c["d_nn_used"], c.get("d_nn_measured")
        return _retune(sysdef, use), measured

    vals = _measure_dnn(a, sysdef, idx)

    if not vals:
        print(f"  d_nn could not be measured; using the registry value "
              f"{sysdef.d_nn} A")
        return sysdef, float("nan")

    measured = float(np.median(vals))
    drift = abs(measured - sysdef.d_nn) / sysdef.d_nn
    print(f"  d_nn measured {measured:.4f} A, median of {len(vals)} frame(s) "
          f"(registry {sysdef.d_nn} A, drift {100*drift:.1f}%)")
    if drift > a.dnn_tol:
        raise SystemExit(
            f"\nSTOP: measured d_nn {measured:.4f} A differs from the "
            f"{sysdef.name} registry value {sysdef.d_nn} A by "
            f"{100*drift:.1f}%, over --dnn-tol {100*a.dnn_tol:.0f}%.\n"
            f"Either --system is wrong for this trajectory, or this state "
            f"point really has a different first shell (expected for the ab "
            f"initio HDL/LDL water). Pass --d-nn {measured:.3f} to accept it "
            f"deliberately, and say so in the SI.")

    if len(vals) < a.dnn_frames:
        print(f"  !! only {len(vals)} frame(s) available to measure d_nn "
              f"(--dnn-frames {a.dnn_frames}). This happens when WINDOW is not "
              f"much larger than STRIDE, so the probe window holds almost no "
              f"kept frames. A one-frame median is noisy; widen WINDOW or pin "
              f"--d-nn.")
    out = _retune(sysdef, measured)
    _write_dnn(a, out, measured=measured, n=len(vals), pinned=False)
    return out, measured


def _write_dnn(a, out, *, measured, n, pinned):
    """Record the cutoff decision next to the features. Every window of a
    trajectory reads this file, so it is the thing that guarantees one d_nn --
    and therefore one set of cutoffs -- across the whole trajectory.

    CONCURRENCY. Ten windows of a trajectory run at once and all of them call
    this. Two rules follow, both learned the hard way:

      * the scratch file must be per-process. A shared `<file>.tmp` means
        window A renames it away while window B is still holding that name, and
        B's os.replace dies with FileNotFoundError. That took out 4 of 6 windows
        on 11 Sep -- the failure looked like a descriptor bug and was a
        temp-name collision.
      * failing to write METADATA must never fail the WINDOW. The features are
        the product; this file is a note about them, and every window writes the
        same content anyway.
    """
    if not a.dnn_file:
        return
    payload = {"system": out.name, "input": os.path.abspath(a.input),
               "d_nn_used": out.d_nn, "d_nn_measured": measured,
               "pinned": pinned, "n_probe_frames": n}

    # already correct? then say nothing and touch nothing -- the common case
    # for the 9 windows that did not decide the value.
    try:
        with open(a.dnn_file) as fh:
            cur = json.load(fh)
        if abs(float(cur.get("d_nn_used", -1)) - out.d_nn) < 1e-12:
            print(f"  d_nn {out.d_nn:.4f} A already recorded in {a.dnn_file}")
            return
    except Exception:
        pass

    tmp = f"{a.dnn_file}.tmp.{os.getpid()}"
    try:
        d = os.path.dirname(os.path.abspath(a.dnn_file))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, a.dnn_file)
        print(f"  d_nn {'pinned' if pinned else 'cached'} -> {a.dnn_file}")
    except Exception as exc:
        print(f"  note: could not record d_nn in {a.dnn_file} "
              f"({type(exc).__name__}: {exc}). Another window is writing the "
              f"same value; continuing.")
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _retune(sysdef, d_nn):
    if d_nn == sysdef.d_nn:
        return sysdef
    return S.System(name=sysdef.name, d_nn=d_nn, fmt=sysdef.fmt,
                    symbol=sysdef.symbol, mass=sysdef.mass,
                    energy_column=sysdef.energy_column,
                    type_map=sysdef.type_map)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, choices=sorted(S.REGISTRY))
    p.add_argument("--input", required=True,
                   help="dump file, extxyz file, or listing (see --kind)")
    p.add_argument("--kind", default="dump",
                   choices=("dump", "extxyz", "filelist"))
    p.add_argument("--root", default=None,
                   help="filelist only: resolve relative paths against this "
                        "directory instead of the listing's own directory")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=None,
                   help="exclusive; default = end of trajectory")
    p.add_argument("--d-nn", type=float, default=None,
                   help="override the measured first-neighbour distance")
    p.add_argument("--dnn-tol", type=float, default=0.03,
                   help="stop if measured d_nn differs from the registry "
                        "value by more than this fraction (default 0.03)")
    p.add_argument("--dnn-file", default=None,
                   help="cache file for this trajectory's d_nn. If it exists, "
                        "its value is USED and nothing is re-measured; if not, "
                        "the measured value is written there. Point every "
                        "window of one trajectory at the same file so they "
                        "cannot end up with different cutoffs.")
    p.add_argument("--dnn-frames", type=int, default=5,
                   help="frames to median over when measuring d_nn")
    p.add_argument("--dnn-check", action="store_true",
                   help="with --d-nn, ALSO measure d_nn and report how far the "
                        "pinned value sits from this trajectory. The pin is "
                        "still what gets used. run_featurize.sh sets this on "
                        "the probe window only, so it costs 5 frames per "
                        "trajectory, not per window.")
    p.add_argument("--energy-column", default=None,
                   help="dump column holding the per-atom potential energy, "
                        "when it is not named c_pe. `compute 1 all pe/atom` "
                        "dumped as c_1 is the common case: pass "
                        "--energy-column c_1. Without this, a dump whose only "
                        "non-standard column is a compute is guessed at (and "
                        "the guess is announced and sanity-checked); a dump "
                        "with several is an error.")
    p.add_argument("--allow-missing-energy", action="store_true")
    p.add_argument("--soap-sigma", type=float, default=1.0,
                   help="SOAP Gaussian width in A. 1.0 is the DScribe default "
                        "and what the published run used; sweeping it is the "
                        "point of the sigma study.")
    p.add_argument("--soap-rbf", default="gto", choices=("gto", "polynomial"))
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "float16"),
                   help="storage dtype for the blocks. float16 halves the disk "
                        "(4.4 -> 2.2 MiB per 4,096-atom frame) at a measured "
                        "cost of <=1.2%% of a column standard deviation, worst "
                        "case, on real mW blocks -- typical 0.1%%, no overflow, "
                        "nothing but a handful of ~1e-8 SOAP components lost. "
                        "It is still a one-way loss on primary data, and it "
                        "does NOT solve stage 2's memory (that needs "
                        "POOL_STRIDE, which is a 10-20x lever, not 2x). "
                        "Default float32.")
    p.add_argument("--overwrite", action="store_true",
                   help="re-featurise frames whose .npy already exists")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true",
                   help="skip the environment banner and the per-step timings")
    a = p.parse_args(argv)

    if not a.quiet and not a.dry_run:
        _install_crash_handlers()
        _print_environment(a)

    sysdef = S.get(a.system)
    n_total = _n_frames(a.kind, a.input, a.root)
    end_frame = n_total if a.frame_end is None else min(a.frame_end, n_total)
    idx = [k for k in range(a.frame_start, end_frame) if k % a.stride == 0]

    print(f"{a.input}")
    print(f"  kind {a.kind}   frames {n_total}   window [{a.frame_start}:{end_frame})"
          f"   stride {a.stride}   -> {len(idx)} frames")
    if not idx:
        print("  nothing to do for this window")
        return 0
    if a.dry_run:
        print(sysdef.describe())
        return 0

    os.makedirs(a.out, exist_ok=True)
    want = set(idx)

    # One sequential scan of the dump gives every frame's byte offset, cached
    # next to the features so it is built once per trajectory, not once per
    # window. Without it, window k re-parses everything before it.
    offsets = None
    if a.kind == "dump":
        cache = os.path.join(a.out, "_meta", "frame_index.json")
        offsets = lammps.frame_offsets(a.input, cache=cache)

    # ---- d_nn: decided ONCE per trajectory, then cached --------------------
    # Measuring per window is a trap. g(r)'s first peak jitters by a bin or two
    # between frames, so two windows of the SAME trajectory can land on
    # different cutoffs and silently produce an inhomogeneous feature matrix.
    # The cache file makes the first window decide and every other window obey.
    sysdef, measured = _resolve_dnn(a, sysdef, idx)
    print(sysdef.describe(), flush=True)

    written, skipped, t0 = [], 0, time.time()
    for k, atoms, energy in _iter_frames(
            a.kind, a.input, sysdef, root=a.root,
            allow_missing_energy=a.allow_missing_energy,
            start=a.frame_start, offsets=offsets,
            energy_column=a.energy_column, stop=end_frame, keep=want):
        if k >= end_frame:
            break
        if k not in want:
            continue

        path = os.path.join(a.out, f"frame_{k:06d}.npy")
        if os.path.exists(path) and not a.overwrite:
            skipped += 1
            continue

        if not written and not skipped:
            # Say it once. A tilted cell takes the scipy Voronoi path, because
            # pyscal3's triclinic branch raises; nothing else about the frame
            # changes. Worth seeing in the log, since it is invisible otherwise.
            cell = np.asarray(atoms.get_cell(), dtype=float)
            if D._is_triclinic(cell):
                tilts = ", ".join(f"{t:.3f}"
                                  for t in cell[np.tril_indices(3, -1)])
                print(f"  cell is TRICLINIC (tilts {tilts} A) -- Voronoi "
                      f"volumes via scipy, not pyscal3", flush=True)
            else:
                print(f"  cell is orthorhombic "
                      f"({cell[0,0]:.3f} x {cell[1,1]:.3f} x {cell[2,2]:.3f} A)",
                      flush=True)

        # "starting" BEFORE the work, so a window killed mid-frame still names
        # the frame it died on. A log whose last line is a completed frame tells
        # you nothing about which frame was fatal.
        print(f"  frame {k:>6} start  n_atoms {len(atoms):>6}  "
              f"RSS {_rss_mb():.0f} MB", flush=True)
        timings = {}
        try:
            block = featurize_frame(atoms, energy, sysdef, timings=timings,
                                soap_sigma=a.soap_sigma, soap_rbf=a.soap_rbf,
                                    energy_column=a.energy_column)
        except MemoryError:
            lim = _mem_limit_mb()
            print(f"  !! MemoryError on frame {k} after "
                  f"{', '.join(timings) or 'no'} step(s). RSS "
                  f"{_rss_mb():.0f} MB, peak {_peak_rss_mb():.0f} MB"
                  + (f", cgroup limit {lim:.0f} MB" if lim else "")
                  + ".\n     Lower CORES in run_featurize.sh (each window is a"
                    " whole copy of the\n     descriptor stack) or raise the"
                    " job's mem.", flush=True)
            raise
        except Exception as exc:
            print(f"  !! frame {k} failed in the "
                  f"{list(timings)[-1] + ' -> next' if timings else 'first'} "
                  f"step: {type(exc).__name__}: {exc}", flush=True)
            raise

        if not written and not np.isfinite(block[:, -1]).any():
            # This used to warn and continue. It warned into a per-window log,
            # 140 blocks of NaN energy were written, and stage 2 then fitted a
            # target scaler that came out `mean [31.5031 nan] sd [2.47821 nan]`
            # and trained a property head against it. A missing regression
            # target is a data problem, and the run has to stop on it.
            if not a.allow_missing_energy:
                raise SystemExit(
                    f"\nSTOP: the energy column is non-finite for every atom "
                    f"of frame {k}.\nNothing in this trajectory carries a "
                    f"per-atom potential energy, so the VAE's second "
                    f"regression target would be fabricated.\n"
                    f"  * LAMMPS dump: add `compute pe all pe/atom` and dump "
                    f"it, or name the column with --energy-column;\n"
                    f"  * extxyz: write a per-atom `energies` array (ASE puts "
                    f"it in calc.results on read, which is checked);\n"
                    f"  * or pass --allow-missing-energy to proceed knowing "
                    f"the column is zeros -- and then train with "
                    f"--lambda-pred 0 so the property head is OFF rather than "
                    f"fitted to nothing, and say so in the SI.")
            print("\n  !! the energy column is entirely non-finite for frame "
                  f"{k}, and --allow-missing-energy was given. The energy "
                  "column of this dataset is NOT data. Train with "
                  "--lambda-pred 0 and do not report a property-prediction "
                  "loss for it.\n", flush=True)
        if block.shape[1] != C.LAYOUT_281.n_cols:
            raise SystemExit(
                f"frame {k}: {block.shape[1]} columns, expected "
                f"{C.LAYOUT_281.n_cols}. Check n_max/l_max and the cutoff count.")
        n_used = sum(1 for x in atoms.get_chemical_symbols()
                     if x == sysdef.symbol)
        if block.shape[0] != n_used:
            raise SystemExit(
                f"frame {k}: {block.shape[0]} rows for {n_used} "
                f"{sysdef.symbol} atoms -- pyscal3 ghost atoms leaked into "
                f"the matrix.")

        tmp = f"{path}.tmp.{os.getpid()}.npy"         # atomic, and per-process
        np.save(tmp, block.astype(a.dtype))           # leaves no half file
        os.replace(tmp, path)
        written.append(path)
        el = time.time() - t0
        steps = "  ".join(f"{k2} {v[0]:.1f}s/{v[2]:+.0f}MB"
                          for k2, v in timings.items())
        rss_before = _rss_mb()
        del block                                    # then break the cycles
        _reclaim()
        print(f"  frame {k:>6} done   {el:7.1f}s cum "
              f"({el/len(written):.2f} s/frame)  RSS {rss_before:.0f} -> "
              f"{_rss_mb():.0f} MB after reclaim  peak {_peak_rss_mb():.0f} MB\n"
              f"           {steps}", flush=True)

    meta = {
        "system": sysdef.name,
        "input": os.path.abspath(a.input),
        "kind": a.kind,
        "d_nn_used": sysdef.d_nn,
        "d_nn_measured": measured,
        "soap_rcut": sysdef.soap_rcut,
        "stein_cutoffs": list(sysdef.stein_cutoffs),
        "ring_cutoff": sysdef.ring_cutoff,
        "coord_cutoff": sysdef.coord_cutoff,
        # record what was USED. A literal here would repeat the pca_umap.py
        # mistake: metadata that names a value the run did not have.
        "soap": {"n_max": C.SOAP_N_MAX, "l_max": C.SOAP_L_MAX,
                 "sigma": a.soap_sigma, "rbf": a.soap_rbf, "periodic": True},
        "layout": "281",
        "window": [a.frame_start, end_frame],
        "stride": a.stride,
        "frames_written": len(written),
        "frames_skipped_existing": skipped,
        "n_frames_total": n_total,
    }
    mdir = os.path.join(a.out, "_meta")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(
            mdir, f"featurization_f{a.frame_start:06d}-{end_frame:06d}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n{len(written)} written, {skipped} already present -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
