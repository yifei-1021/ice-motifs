"""
Datasets and splits.

The split is the one thing here that is deliberately *not* a faithful copy of the
original. `procdata` split by row index (`arange(N) % 5 == 0`), and because rows are
laid out frame-by-frame that puts atoms from every frame into both train and test:
all 141 frames appeared on both sides, none was held out. Near-identical train/test
loss then follows by construction, so it cannot be evidence against overfitting -
which is exactly Reviewer 1's objection.

`make_split` therefore offers three modes and defaults to the honest one:

    "frame"  hold out whole frames               <- default
    "block"  hold out a contiguous tail          (e.g. late-time behaviour)
    "index"  the original `% 5` interleave       (reproduces published numbers)

Use mode="index" when you need to reproduce the published table; use "frame" when you
want a number you can defend.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import columns as C


class ArrayDataset(Dataset):
    def __init__(self, array):
        self.data = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def loader(array, batch_size=1024, shuffle=True, num_workers=0):
    return DataLoader(ArrayDataset(array), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


def clean(array, verbose=True):
    """Replace non-finite values with 0 and report how many there were.

    The original `removeNan` computed a mask and then ignored it, calling nan_to_num -
    so rows were zero-filled, not dropped, despite the name. Same behaviour here, but
    it says so and it tells you the count.
    """
    a = np.asarray(array, dtype=np.float32)
    bad = int((~np.isfinite(a)).sum())
    if bad and verbose:
        rows = int((~np.isfinite(a)).any(axis=1).sum())
        print(f"  clean(): {bad} non-finite values in {rows} rows -> replaced with 0.0")
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def make_split(n_rows, mode="frame", atoms_per_frame=None,
               train=0.133, val=0.067, seed=0):
    """Return (train_idx, val_idx, test_idx)."""
    idx = np.arange(n_rows)

    if mode == "index":
        tr_pool = idx[idx % 5 == 0]
        rest = idx[idx % 5 != 0]
        v = tr_pool[np.arange(tr_pool.size) % 3 == 0]
        t = tr_pool[np.arange(tr_pool.size) % 3 != 0]
        return t, v, rest

    if mode == "frame":
        if not atoms_per_frame:
            raise ValueError("mode='frame' needs atoms_per_frame")
        n_frames = int(np.ceil(n_rows / atoms_per_frame))
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_frames)
        n_tr = max(1, int(round(train * n_frames)))
        n_va = max(1, int(round(val * n_frames)))
        groups = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va],
                  "test": order[n_tr + n_va:]}
        out = []
        for k in ("train", "val", "test"):
            frames = np.sort(groups[k])
            m = np.zeros(n_rows, dtype=bool)
            for f in frames:
                m[f * atoms_per_frame:(f + 1) * atoms_per_frame] = True
            out.append(idx[m])
        return tuple(out)

    if mode == "block":
        n_tr, n_va = int(train * n_rows), int(val * n_rows)
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    raise ValueError(f"unknown split mode {mode!r}; use 'frame', 'block' or 'index'")


def split_report(name, tr, va, te, n_rows, atoms_per_frame=None):
    print(f"  split '{name}': train {tr.size} ({100*tr.size/n_rows:.1f}%)  "
          f"val {va.size} ({100*va.size/n_rows:.1f}%)  "
          f"test {te.size} ({100*te.size/n_rows:.1f}%)")
    if atoms_per_frame:
        f = lambda a: set((a // atoms_per_frame).tolist())
        ftr, fte = f(tr), f(te)
        shared = len(ftr & fte)
        print(f"    frames: train {len(ftr)}, test {len(fte)}, shared {shared}"
              + ("   <- test is NOT independent" if shared else "   <- disjoint"))


def _block_shapes(path, files):
    """(rows, ncols, dtypes) from the .npy HEADERS, one descriptor at a time.

    See the note in load_features: holding one open memory-map per block runs
    out of file descriptors at ~100k blocks, and the shape does not need one.
    """
    import os

    import numpy.lib.format as _fmt

    rows, ncols, dtypes = [], set(), set()
    for f in files:
        with open(os.path.join(path, f), "rb") as fh:
            ver = _fmt.read_magic(fh)
            rd = (_fmt.read_array_header_1_0 if ver == (1, 0)
                  else _fmt.read_array_header_2_0)
            shape, _fortran, dtype = rd(fh)
        if len(shape) != 2:
            raise SystemExit(f"{f} is not 2-D; got shape {shape}")
        rows.append(int(shape[0]))
        ncols.add(int(shape[1]))
        dtypes.add(dtype)
    if len(ncols) != 1:
        raise SystemExit(
            f"blocks in {path} have differing widths {sorted(ncols)}; "
            "they were not all produced by the same featurisation run")
    return rows, ncols.pop(), dtypes


def load_features(path, layout=None, verbose=True):
    """Load one .npy matrix, or a DIRECTORY of per-frame .npy blocks.

    The directory form is what stage 1 writes for the new systems: one block
    per frame, named frame_%06d.npy. Blocks are concatenated in SORTED FILENAME
    ORDER, which is frame order, because every downstream analysis that looks
    at time (transition matrices, motif lifetimes) assumes row order follows
    the trajectory.
    """
    import os

    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".npy"))
        if not files:
            raise SystemExit(f"no .npy blocks in {path}")
        # Read the headers first and PREALLOCATE. The obvious
        # `np.vstack([np.load(f) for f in files])` holds the whole list and the
        # result at the same time, so its peak is 2x the data -- which is what
        # killed the heterogeneous job: 17.7M rows x 281 is 18.5 GB, and 37 GB
        # of transient peak did not fit in a 64 GB job. Filling a preallocated
        # array block by block peaks at the data plus one block.
        # ... and read those headers WITHOUT mmap. One open memory-map per
        # block means one file descriptor per block, all held at once: fine at
        # 1,923 blocks, "OSError: [Errno 24] Too many open files" at the
        # 100,125 of the Gartner set. The shape is in a 128-byte header.
        rows, ncol, dtypes = _block_shapes(path, files)
        # float16 is a STORAGE choice (featurize.py --dtype float16). Nothing
        # downstream should compute in half precision: the scalers, UMAP and
        # HDBSCAN would all quietly upcast anyway, and a float16 accumulator
        # anywhere is a silent accuracy bug. Promote once, here, where it is
        # visible in the log.
        out_dtype = np.float32 if np.float16 in dtypes else sorted(
            dtypes, key=str)[0]
        total = int(sum(rows))
        if verbose:
            print(f"  loading {len(files)} blocks from {path}: "
                  f"({total:,}, {ncol}) = "
                  f"{total * ncol * np.dtype(out_dtype).itemsize / 2**30:.2f} GB"
                  + (f"  [stored as {sorted(str(d) for d in dtypes)}, "
                     f"promoted to float32]" if np.float16 in dtypes else ""))
        X = np.empty((total, ncol), dtype=out_dtype)
        at = 0
        for f, n in zip(files, rows):
            b = np.load(os.path.join(path, f))
            X[at:at + n] = b
            at += n
            del b
    else:
        X = np.load(path, mmap_mode=None)
        if verbose:
            print(f"  loaded {path}: {X.shape}")
    layout = layout or C.infer(X)
    layout.check(X)
    if verbose:
        print(f"    layout {layout.name}")
    return X, layout


def block_rows(path, files=None):
    """Rows in each .npy block, read from the HEADERS alone.

    The headers are 128 bytes each, so this does not load, map or even fully
    open the data. That matters at scale: the Gartner pool is 100,125 blocks,
    and loading one memory-map per block to ask for its shape takes minutes
    where this takes seconds. `files` selects a subset, in the order given.
    """
    import os

    files = (sorted(f for f in os.listdir(path) if f.endswith(".npy"))
             if files is None else list(files))
    return _block_shapes(path, files)[0]


def frames_in(path):
    """Rows per frame for a directory of blocks, or None for a single file."""
    import os

    if not os.path.isdir(path):
        return None
    return block_rows(path)
