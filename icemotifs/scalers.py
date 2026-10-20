"""
Standardisation, and how it is stored.

The original code pickled a `util.TorchStandardScaler` holding CUDA tensors. That is
fragile twice over: unpickling needs a module literally named `util` on sys.path, and
the CUDA tensors fail to load on a CPU-only machine unless torch.load is patched. Both
bite when you move laptops.

So: the class is kept (old pickles still load, see `load`) but new scalers are written
as plain .npz of mean/scale. Nothing to unpickle, nothing device-specific.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch


class TorchStandardScaler:
    """Kept name-compatible with the original `util.TorchStandardScaler`."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.shape = self.mean.shape

    # the original guarded against division by zero only implicitly; make it explicit
    def transform(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse_transform(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def save(self, path):
        np.savez(path, mean=self.mean.cpu().numpy(), scale=self.std.cpu().numpy())

    def __repr__(self):
        return f"TorchStandardScaler(dim={len(self.mean)})"


def fit(array, eps=1e-8):
    """Per-column mean and sd, ignoring non-finite entries.

    Identical to `a.mean(0)` / `a.std(0)` whenever every value is finite, so
    nothing that ran before changes. The guard matters because the old
    `std[std < eps] = 1.0` does NOT catch NaN -- `NaN < eps` is False -- so an
    all-NaN column wrote `mean=nan, sd=nan` straight into scaler_targets.npz
    and every scaled value downstream became NaN silently. That is exactly the
    64,000-particle energy bug, and it should be impossible rather than caught
    by eye.
    """
    a = np.asarray(array, dtype=np.float64)
    ok = np.isfinite(a)
    if ok.all():
        mean, std = a.mean(axis=0), a.std(axis=0)
    else:
        n = ok.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, np.nansum(np.where(ok, a, 0.0), axis=0)
                            / np.maximum(n, 1), 0.0)
            var = np.where(n > 0,
                           np.nansum(np.where(ok, (a - mean) ** 2, 0.0), axis=0)
                           / np.maximum(n, 1), 0.0)
        std = np.sqrt(var)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 1.0)
    std[std < eps] = 1.0            # constant columns would otherwise produce inf
    return TorchStandardScaler(mean, std)


def load(path):
    """Load either a new .npz scaler or a legacy pickle (CUDA tensors included)."""
    p = str(path)
    if p.endswith(".npz"):
        z = np.load(p)
        return TorchStandardScaler(z["mean"], z["scale"])

    # legacy pickle: the stored tensors may be on CUDA, and the pickle references
    # `util.TorchStandardScaler`, so provide that name and force a CPU map_location.
    import sys
    import types

    if "util" not in sys.modules:
        shim = types.ModuleType("util")
        shim.TorchStandardScaler = TorchStandardScaler
        sys.modules["util"] = shim

    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "map_location": "cpu", "weights_only": False})
    try:
        with open(p, "rb") as fh:
            obj = pickle.load(fh)
    finally:
        torch.load = orig

    mean = getattr(obj, "mean", None)
    std = getattr(obj, "std", getattr(obj, "scale_", None))
    if mean is None or std is None:                 # sklearn StandardScaler
        mean, std = obj.mean_, obj.scale_
    return TorchStandardScaler(np.asarray(torch.as_tensor(mean).cpu()),
                               np.asarray(torch.as_tensor(std).cpu()))


def fit_blocks(array, layout):
    """Fit one scaler per block: Steinhardt, SOAP, regression targets."""
    return (fit(array[:, layout.stein]),
            fit(array[:, layout.soap]),
            fit(array[:, layout.targets_slice]))


def save_blocks(scalers, outdir, names=("steinhardt", "soap", "targets")):
    import os
    paths = []
    for s, n in zip(scalers, names):
        p = os.path.join(outdir, f"scaler_{n}.npz")
        s.save(p)
        paths.append(p)
    return paths


def load_blocks(outdir, names=("steinhardt", "soap", "targets")):
    import os
    return tuple(load(os.path.join(outdir, f"scaler_{n}.npz")) for n in names)
