"""
The inference half of stage 3: a trained VAE over a feature matrix.

The stage that calls this is `icemotifs encode`
(`icemotifs/pipeline/encode_run.py`), which resolves the model, the scalers
and the cutoff check and writes the run layout. `encode_array` returns, for
N input rows:

    Z.npy           (N, dim_z)   sampled latent, z = mean + eps*sigma
    mean.npy        (N, dim_z)   latent mean          <- use this for clustering
    log_var.npy     (N, dim_z)   latent log-variance
    log_sigma_norm.npy (N,)      ||log sigma||, the local-disorder measure
    recon.npy       (N, n_cols)  reconstruction, un-scaled, same layout as the input
    metrics.json                 per-block MSE in scaled space

Two things worth knowing:

  * Cluster on `mean`, not `Z`. `Z` includes the reparameterisation noise, so it is
    not reproducible run to run; `mean` is deterministic under eval().
  * ||log sigma||. SI Section 4 calls it "the root-mean-square" but writes a
    root-SUM-square (no 1/dim(Z)); with dim(Z)=40 those differ by sqrt(40) ~ 6.3.
    Both are written here - `log_sigma_norm.npy` is the norm as the formula has it,
    and metrics.json records the RMS too - so plots can state which they use.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from . import model as M
from . import scalers as S


@torch.no_grad()
def encode_array(model, X, layout, scal=None, batch_size=4096, device="cpu",
                 deterministic=True, want_recon=True, seed=123456):
    """Run the model over X and return (Z, mean, log_var, recon, metrics).

    `deterministic=True` writes the latent MEAN into Z, but it does not make
    the METRICS deterministic: VAE2.forward decodes from a reparameterised
    sample (`mean + randn_like(std) * std`), so the reconstruction and the
    property prediction carry that noise whatever Z is set to. Two calls on
    the same rows returned mse_steinhardt 1.012052 and 1.013091 -- 0.1% apart,
    on a number the revision quotes to six figures.

    So `seed` fixes the sampler, and restores the caller's RNG state on the way
    out. The decode stays faithful to what the published run did; it just stops
    giving a different answer each time it is asked. Pass seed=None for the old
    unseeded behaviour.
    """
    model.eval().to(device)
    rng_state = None
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
    try:
        return _encode_array(model, X, layout, scal, batch_size, device,
                             deterministic, want_recon, seed)
    finally:
        if rng_state is not None:
            torch.set_rng_state(rng_state)


@torch.no_grad()
def _encode_array(model, X, layout, scal, batch_size, device,
                  deterministic, want_recon, seed):
    s_st, s_so, s_tg = (scal if scal is not None else (model.scaler1, model.scaler2, None))
    for s in (s_st, s_so):
        s.to(device)
    if s_tg is not None:
        s_tg.to(device)

    Z, MU, LV, RE = [], [], [], []
    mse = nn.MSELoss(reduction="sum")
    acc = dict(stein=0.0, soap=0.0, pred=0.0)
    # Per-target, not just the sum. With the published regrsca.pkl the two
    # targets are on scales 800x apart, so a summed prediction MSE is a
    # Voronoi number wearing an energy label.
    acc_t = np.zeros(len(layout.targets), dtype=np.float64)
    n = 0

    for i in range(0, len(X), batch_size):
        b = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        x1, x2 = b[:, layout.stein], b[:, layout.soap]
        (_, _), (mean, log_var), (z, r1, r2, py) = model(x1, x2)
        if deterministic:
            z = mean                     # no reparameterisation noise
        acc["stein"] += float(mse(r1, s_st.transform(x1)))
        acc["soap"] += float(mse(r2, s_so.transform(x2)))
        blocks = [s_st.inverse_transform(r1), s_so.inverse_transform(r2)]
        if s_tg is not None:
            y = b[:, layout.targets_slice]
            d2 = (py - s_tg.transform(y)) ** 2
            acc["pred"] += float(d2.sum())
            acc_t += d2.sum(dim=0).cpu().numpy()
            blocks.append(s_tg.inverse_transform(py))
        if want_recon:
            RE.append(torch.cat(blocks, dim=1).cpu().numpy())
        Z.append(z.cpu().numpy())
        MU.append(mean.cpu().numpy())
        LV.append(log_var.cpu().numpy())
        n += len(b)

    Z, MU, LV = (np.vstack(a) for a in (Z, MU, LV))
    # `recon.npy` is (n_rows, 281) -- as big as the input. Building it costs
    # twice that in transient during the vstack, and only icemotifs.encode's
    # CLI ever writes it, so callers that do not want it say so and pay
    # nothing.
    RE = np.vstack(RE) if want_recon else None
    d = MU.shape[1]
    metrics = {
        "n_rows": int(n), "dim_z": int(d),
        "mse_steinhardt": acc["stein"] / (n * layout.n_stein),
        "mse_soap": acc["soap"] / (n * layout.n_soap),
        "mse_prediction": (acc["pred"] / (n * len(layout.targets))) if s_tg is not None else None,
        "decode_seed": seed,
    }
    if s_tg is not None:
        metrics["mse_per_target"] = {name: float(v / n)
                                     for name, v in zip(layout.targets, acc_t)}
    return Z, MU, LV, RE, metrics


def log_sigma_norm(log_var, as_rms=False):
    """||log sigma||. log_var is log(sigma^2), so log(sigma) = log_var/2."""
    ls = 0.5 * log_var
    sq = (ls ** 2).sum(axis=1)
    return np.sqrt(sq / ls.shape[1]) if as_rms else np.sqrt(sq)
