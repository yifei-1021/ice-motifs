"""
The training half of stage 2: the VAE over a .npy feature matrix.

`icemotifs train` (`icemotifs/pipeline/train_vae.py`) is the stage, and it owns
the whole run: the published hyperparameters, the scalers, the argv and the epoch
loop. What is left here is the arithmetic that loop calls -- the beta schedule,
the KLD and one epoch's forward/backward pass -- so it can be imported and
tested without a run directory.

Loss, as published:

    L = l1*MSE(stein) + l2*MSE(soap) + l3*MSE(targets) + beta*KLD

with l1 = l2 = 1, l3 = 0.1 and beta cycled 0 -> 1e-3 over 10,000-iteration cycles.

Two notes on faithfulness:

  * l3. The original computed `lambda3 = 1/beta` from a `--BETA` CLI argument whose
    default was 4, giving 0.25 - not the published 0.1, which needs --BETA 10. Here
    l3 is `--lambda-pred`, default 0.1, so the default reproduces the paper.
  * KLD. `calculateKLD` took torch.mean over batch *and* latent dims; SI Eq. (S16)
    writes a sum over dim(Z). They differ by a factor dim(Z) = 40, which beta absorbs.
    `--kld-reduction` selects: "mean" (the code that produced the published run) or
    "sum" (what the SI equation says). Default "mean".
"""

from __future__ import annotations

import torch
import torch.nn as nn


def cyclical_beta(iteration, beta_max=1e-3, interval=10000):
    """Ramp 0 -> beta_max over the first half of each cycle, then hold.

    interval <= 0 means CONSTANT beta = beta_max. The PUBLISHED run used the
    CYCLICAL branch: verified against the training log, where the logged
    beta*KLD term is a sawtooth with a 10,000-iteration period whose phase
    explains 93.6% of its variance. An earlier version of this docstring said
    the opposite; that was a misreading and is withdrawn.
    """
    if interval <= 0:
        return beta_max
    i = iteration % interval
    half = interval // 2
    return beta_max if i > half else (i / half) * beta_max


def kld(mean, log_var, reduction="mean"):
    per = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    return per.mean() if reduction == "mean" else per.sum(dim=1).mean()


def run_epoch(model, loader, layout, scal, opt=None, *, lambdas=(1.0, 1.0, 0.1),
              beta_max=1e-3, beta_interval=10000, kld_reduction="mean",
              it0=0, device="cpu"):
    train = opt is not None
    model.train(train)
    s_st, s_so, s_tg = scal
    l1, l2, l3 = lambdas
    mse = nn.MSELoss()
    tot = dict(loss=0.0, stein=0.0, soap=0.0, pred=0.0, kld=0.0)
    n = 0
    it = it0

    for batch in loader:
        batch = batch.to(device)
        x1 = batch[:, layout.stein]
        x2 = batch[:, layout.soap]
        y = batch[:, layout.targets_slice]
        if not torch.isfinite(batch).all():
            continue

        with torch.set_grad_enabled(train):
            (_, _), (mean, log_var), (_, r1, r2, py) = model(x1, x2)
            l_st = mse(r1, s_st.transform(x1))
            l_so = mse(r2, s_so.transform(x2))
            l_pr = mse(py, s_tg.transform(y))
            l_kl = kld(mean, log_var, kld_reduction)
            beta = cyclical_beta(it, beta_max, beta_interval)
            loss = l1 * l_st + l2 * l_so + l3 * l_pr + beta * l_kl

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1

        for k, v in zip(tot, (loss, l_st, l_so, l_pr, l_kl)):
            tot[k] += float(v)
        n += 1

    return {k: v / max(n, 1) for k, v in tot.items()}, it
