"""
Loading the legacy checkpoints.

`model.last.pth` was written with `torch.save(model, ...)` - a full pickle. Unpickling
it needs modules literally named `model_concatdecoder` (for VAE2) and `util` (for the
TorchStandardScaler instances stored as model attributes), and, on torch >= 2.6,
`weights_only=False`, whose default flipped to True.

`load_legacy` installs those two names, loads, and hands back a model whose scalers
have been moved to CPU. Then convert once:

    from icemotifs import compat, model
    m = compat.load_legacy("model.last.pth")
    model.save(m, "model.pt")        # a state_dict; stop depending on the pickle

Only unpickle checkpoints you trust - `weights_only=False` executes arbitrary code.
"""

from __future__ import annotations

import sys
import types

import torch

from . import model as _model
from .scalers import TorchStandardScaler


def _install_shims():
    if "util" not in sys.modules:
        u = types.ModuleType("util")
        u.TorchStandardScaler = TorchStandardScaler
        sys.modules["util"] = u
    for name in ("model_concatdecoder", "model", "vae"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.VAE2 = _model.VAE2
            m.TorchStandardScaler = TorchStandardScaler
            sys.modules[name] = m


PUBLISHED_SCALER_FILES = ("steinsc.pkl", "soapsca.pkl", "regrsca.pkl")


def load_published_scalers(dirpath, model=None, verbose=True):
    """Load steinsc/soapsca/regrsca.pkl from the released model folder.

    `model`, if given, is a checkpoint loaded by `load_legacy`; its own
    scaler1/scaler2 are then compared against the files. They should be the
    same objects -- if they are not, the folder and the checkpoint are from
    different runs and nothing downstream is comparable to the paper.
    """
    import os

    import numpy as np

    from . import scalers as S

    paths = [os.path.join(dirpath, f) for f in PUBLISHED_SCALER_FILES]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "not the published model folder: missing "
            + ", ".join(os.path.basename(p) for p in missing)
            + f"\n  looked in {dirpath}\n"
            "  it must hold steinsc.pkl, soapsca.pkl and regrsca.pkl "
            "beside model.last.pth")
    scal = tuple(S.load(p) for p in paths)

    if model is not None:
        for i, attr in enumerate(("scaler1", "scaler2")):
            s = getattr(model, attr, None)
            if s is None:
                continue
            a = np.asarray(s.mean.cpu()), np.asarray(s.std.cpu())
            b = np.asarray(scal[i].mean), np.asarray(scal[i].std)
            if (a[0].shape != b[0].shape
                    or not np.allclose(a[0], b[0], atol=1e-5)
                    or not np.allclose(a[1], b[1], atol=1e-5)):
                raise SystemExit(
                    f"{PUBLISHED_SCALER_FILES[i]} does not match the checkpoint's "
                    f"{attr}. The scalers and the model are from different runs; "
                    "encoding with them would not reproduce anything.")
        if verbose:
            print("  scalers match the checkpoint's own scaler1/scaler2")

    if verbose:
        m, sd = np.asarray(scal[2].mean), np.asarray(scal[2].std)
        print(f"  target scaler: mean {np.round(m, 4).tolist()}  "
              f"sd {np.round(sd, 4).tolist()}")
        if len(sd) > 1 and sd[1] > 1.0:
            print("  !! the energy entry of regrsca.pkl was fitted on an array that")
            print("     still held the ice-phase sentinels: sd 23.58 against a real")
            print("     spread of 0.03. Energy is divided by ~800x too much, so the")
            print("     energy half of the prediction MSE is near zero by construction.")
            print("     Read mse_per_target, not the summed number, and do not quote")
            print("     the energy term as evidence of anything.")
    return scal


def load_legacy(path, map_location="cpu"):
    """Load a full-pickle checkpoint and return it with CPU scalers."""
    _install_shims()
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, "map_location": map_location,
                                             "weights_only": False})
    try:
        m = orig(path, map_location=map_location, weights_only=False)
    finally:
        torch.load = orig

    for attr in ("scaler1", "scaler2"):
        s = getattr(m, attr, None)
        if s is not None and hasattr(s, "mean"):
            s.mean = s.mean.cpu()
            s.std = s.std.cpu()
    m.eval()
    return m
