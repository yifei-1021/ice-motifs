"""
The published training configuration, transcribed from the original scripts.

Sources: `model_concatdecoder.py` (VAE2, procdata, processScalers),
`util.py` (TorchStandardScaler, loadScaler), `collect_all.py` /
`collect_test.py` (loss weights, KLD and MSE reductions), and the training log.

Every value here was cross-checked against the released checkpoint, the
released scalers and `logger.log`, not just read off the source. Where the
original does something questionable, this module reproduces it faithfully and
says so, and offers the corrected variant beside it — so "reproduce the paper"
and "do it properly" are both one call away, and never silently confused.

Use it when training a VAE on a NEW dataset and you want the published
settings:

    from icemotifs import published
    cfg = published.CONFIG                       # the hyperparameters
    tr, va, te = published.split(len(X))         # the published split
    scal = published.fit_scalers(X[tr], layout)  # CORRECTED: training rows only
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------- input space

# The cutoffs the released checkpoint was trained at. `water-mlbop` and
# `water-mw` reproduce these exactly (d_nn 2.76 x the reduced cutoffs); any
# other system is measured at the same coordination SHELL but a different
# absolute radius, which is the whole point of the reduced units -- and also
# the reason the published weights do not transfer to it unexamined.
SOAP_RCUT = 6.0
STEIN_CUTOFFS = (3.2, 4.8, 6.0)


def cutoff_report(sysdef, rtol=0.01):
    """(ok, lines) -- can the published checkpoint read `sysdef`'s features?"""
    got_soap = float(sysdef.soap_rcut)
    got_stein = tuple(float(c) for c in sysdef.stein_cutoffs)
    ok_soap = abs(got_soap - SOAP_RCUT) <= rtol * SOAP_RCUT
    ok_stein = all(abs(a - b) <= rtol * b
                   for a, b in zip(got_stein, STEIN_CUTOFFS))
    lines = [f"  published   SOAP r_cut {SOAP_RCUT:.2f} A   "
             f"Steinhardt {', '.join(f'{c:.2f}' for c in STEIN_CUTOFFS)} A",
             f"  {sysdef.name:<11} SOAP r_cut {got_soap:.2f} A   "
             f"Steinhardt {', '.join(f'{c:.2f}' for c in got_stein)} A"
             f"   (d_nn {sysdef.d_nn})"]
    if ok_soap and ok_stein:
        lines.append("  -> same input space. The published weights apply directly.")
    else:
        lines.append("  -> DIFFERENT input space. These descriptors are not the ones")
        lines.append("     the checkpoint was trained on: the same column means a")
        lines.append("     different radius. On 3.2 A features the published model")
        lines.append("     scored a SOAP MSE 1943x worse than on 6.0 A ones, with no")
        lines.append("     error and no warning. Whatever comes out is a transfer")
        lines.append("     experiment, not a measurement, and must be labelled as one.")
    return (ok_soap and ok_stein), lines


def domain_report(X, layout, scal, n_sample=200000, seed=0):
    """How far outside the published standardisation the new rows sit.

    Standardise with the PUBLISHED scalers and look at the per-column mean.
    Zero means the new data is centred where the training data was. A column
    sitting many standard deviations away is extrapolation: the encoder has
    never seen that region and its output is an extension, not a reading.
    """
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(X), n_sample, replace=False)
           if len(X) > n_sample else slice(None))
    S = np.asarray(X[idx], dtype=np.float64)
    lines, worst = [], 0.0
    for name, sl, sc in (("Steinhardt", layout.stein, scal[0]),
                         ("SOAP", layout.soap, scal[1])):
        mean = np.asarray(sc.mean, dtype=np.float64)
        std = np.asarray(sc.std, dtype=np.float64)
        z = np.abs((S[:, sl].mean(axis=0) - mean) / std)
        worst = max(worst, float(z.max()))
        lines.append(f"  {name:<10} |shift| median {np.median(z):.2f} sd, "
                     f"p95 {np.percentile(z, 95):.2f}, max {z.max():.2f}   "
                     f"({int((z > 5).sum())} of {z.size} columns beyond 5 sd)")
    if worst < 2:
        lines.append("  -> well inside the published training distribution.")
    elif worst < 5:
        lines.append("  -> shifted but overlapping. Compare reconstruction error "
                     "against the published values before reading the latent.")
    else:
        lines.append("  -> OUTSIDE the published training distribution. The encoder "
                     "is extrapolating; latent positions are not comparable to the "
                     "published motif map.")
    return worst, lines


# --------------------------------------------------------------------------- model

ARCHITECTURE = dict(
    input_steinhardt=27,
    input_soap=252,
    encoder1=(128, 64, 20),        # 27 -> 128 -> 64 -> 20
    encoder2=(512, 256, 20),       # 252 -> 512 -> 256 -> 20
    latent_layers=1,               # num_latent_layers
    latent=40,                     # VAE2(latent=...); self.latent = 2*40 = 80
    dim_z=40,                      # = self.latent // 2
    decoder_layers=2,              # num_decoder_layers
    decoder_hidden=2048,           # hardcoded as self.hidden1 in VAE2
    predictor_hidden=512,          # HIDDEN_DIM
    regression_targets=2,          # Voronoi volume, potential energy
    activation="tanh",
    dropout=0.10,                  # not applied to the latent layers
)

# --------------------------------------------------------------------------- training

CONFIG = dict(
    optimizer="adam",
    lr=1e-4,
    batch_size=1000,               # per the training log's tensor shapes
    epochs=200,
    seed=123456,                   # torch.manual_seed and np.random.seed
    lambda1=1.0,                   # Steinhardt reconstruction
    lambda2=1.0,                   # SOAP reconstruction
    lambda3=0.1,                   # property prediction (written as 1/10)
    beta_max=1e-3,
    beta_schedule="cyclical",      # 10,000-iteration period, verified in the log
    beta_interval=10000,
    kld_reduction="mean",          # calculateKLD uses torch.mean over ALL elements
    mse_reduction="mean",          # nn.MSELoss(reduction='mean')
    **ARCHITECTURE,
)

# What the original does that you should know about before copying it ----------
KNOWN_ISSUES = """
1. EARLY STOPPING NEVER STOPS. The criterion is evaluated and a "Stopping
   Training at Epoch N" line is logged, but the loop does not break. The
   published run logged it at epochs 65 and 193 and still ran to 200.
   `train_new_system.py` reproduces the full 200 epochs, which is what happened.

2. THE SCALERS WERE FITTED ON THE WHOLE DATASET. processScalers() slices
   `dataset`, not the training split, so standardisation statistics were
   computed over train + validation + test. `fit_scalers` below fits on
   whatever rows you pass it; pass the training rows.

3. THE SPLIT IS NOT FRAME-DISJOINT. It is a stride on the row index, and rows
   follow the trajectory, so every frame contributes to all three sets. Use
   `split_by_block` if you want a genuine held-out set.

4. removeNan() computes a row mask and ignores it, replacing non-finite values
   with 0.0 in RAW space -- which becomes -mean/sd after scaling, a fabricated
   extreme rather than a neutral value. `clean_rows` below drops them instead.

5. THE PROPERTY TARGETS ARE NOT MASKED IN THE SCALER FIT. Rows whose energy is
   the sentinel 1000.0 (reference phases, ignored in the loss) were included
   when regrsca was fitted, which compresses the real energy target to ~0.1% of
   unit variance. `fit_scalers(..., sentinel=1000.0)` masks them.
"""


# --------------------------------------------------------------------------- split

def split_published(n):
    """The published split: keep every 5th row, then every 3rd to validation.

    Reproduces 2,407,738 / 1,203,870 / 14,446,432 for n = 18,058,040 exactly.

    NOT frame-disjoint. Rows follow the trajectory, so a stride of 5 draws
    atoms from every frame into all three sets; the evaluation set is held out
    per local environment, never per configuration. Use `split_by_block` for a
    genuine held-out set.
    """
    idx = np.arange(n)
    keep = idx[idx % 5 == 0]
    test = idx[idx % 5 != 0]
    pos = np.arange(len(keep))
    valid = keep[pos % 3 == 0]
    train = keep[pos % 3 != 0]
    return train, valid, test


def split_by_block(n, n_blocks=141, val_blocks=0.15, test_blocks=0.20, seed=123456):
    """A frame-disjoint alternative: split whole contiguous time blocks.

    This is the split the paper should have used, and the one that can answer
    R1's question about whether the representation generalises. Contiguous
    blocks are safe even without exact frame boundaries: rows follow the
    trajectory, so a block is a narrow time window.
    """
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    n_te = max(1, int(round(test_blocks * n_blocks)))
    n_va = max(1, int(round(val_blocks * n_blocks)))
    te_b, va_b = order[:n_te], order[n_te:n_te + n_va]
    tr_b = order[n_te + n_va:]

    def rows(bs):
        return np.concatenate([np.arange(edges[b], edges[b + 1]) for b in sorted(bs)])

    return rows(tr_b), rows(va_b), rows(te_b)


# --------------------------------------------------------------------------- data

def clean_rows(X, verbose=True):
    """Drop non-finite rows, rather than substituting 0.0 into raw space."""
    ok = np.isfinite(X).all(axis=1)
    if verbose and not ok.all():
        print(f"  dropping {int((~ok).sum())} non-finite rows of {len(X)}")
    return X[ok], ok


def fit_scalers(X, layout, sentinel=1000.0, verbose=True,
                allow_dead_targets=False):
    """Fit the three scalers on the rows given, masking sentinel targets.

    Two differences from the original `processScalers`, both deliberate:
      * fitted on whatever you pass (pass the TRAINING rows, not everything);
      * rows whose target equals `sentinel` are excluded from the TARGET fit
        only -- their descriptors are still real and still count.
        Pass sentinel=None to reproduce the original behaviour exactly.
    """
    out = {}
    for name, sl in (("steinhardt", layout.stein), ("soap", layout.soap)):
        B = X[:, sl]
        out[name] = (B.mean(0), B.std(0))
    # Layout stores the target block as a name tuple, not a slice, so derive it
    # from the descriptor widths rather than assuming column 279.
    t0 = layout.n_stein + layout.n_soap
    T = X[:, t0:t0 + len(layout.targets)]
    if sentinel is not None:
        keep = ~(T == sentinel).any(axis=1)
        if verbose and (~keep).any():
            n = int((~keep).sum())
            print(f"  masking {n} sentinel rows ({100*n/len(T):.3f}%) from the "
                  f"target scaler -- the original did NOT do this")
        T = T[keep]

    # Non-finite targets must be masked too, not just the sentinel. A single
    # NaN makes mean and sd NaN for that column, every scaled value NaN, and
    # the run continues -- which is how the 64,000-particle set trained a
    # property head against a column that was NaN for all 8,960,000 rows and
    # reported `target scaler: mean [31.5031 nan]  sd [2.47821 nan]` without
    # stopping. A target that is missing everywhere is a data problem, and it
    # has to be raised here rather than absorbed.
    finite = np.isfinite(T)
    dead = [layout.targets[j] for j in range(T.shape[1])
            if not finite[:, j].any()]
    live = [j for j in range(T.shape[1]) if finite[:, j].any()]
    if dead and allow_dead_targets:
        # The property head is OFF (lambda3 == 0), so nothing regresses this
        # column and a scaler for it is never applied to a loss. Warn loudly,
        # neutralise the dead column to (0, 1) so it can never inject NaN, and
        # keep the live columns' real statistics -- masking rows on a column
        # that is dead everywhere would otherwise drop every row and destroy
        # the perfectly good Voronoi column beside it.
        print(f"\n  !! target(s) {dead} are non-finite in every one of the "
              f"{len(T):,} rows.\n     lambda3 = 0, so the property head is "
              f"OFF and this is allowed: the column is\n     neutralised to "
              f"mean 0 sd 1 and MUST NOT be reported as a prediction loss.\n",
              flush=True)
        mean = np.zeros(T.shape[1])
        sd = np.ones(T.shape[1])
        if live:
            L = T[:, live]
            keep_rows = np.isfinite(L).all(axis=1)
            L = L[keep_rows]
            mean[live] = L.mean(0)
            sd[live] = L.std(0)
        sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        out["targets"] = (mean, sd)
        if verbose:
            print(f"  target scaler: mean {np.round(mean, 5)}  "
                  f"sd {np.round(sd, 5)}")
        return out
    if dead:
        raise SystemExit(
            f"\nSTOP: target(s) {dead} are non-finite in every one of the "
            f"{len(T):,} rows used to fit the scaler.\n"
            f"Nothing carries that quantity in this dataset, so the model "
            f"cannot regress it and a scaler cannot be fitted for it.\n"
            f"  * for a LAMMPS dump, dump a per-atom energy "
            f"(`compute pe all pe/atom`) or name the column that holds it "
            f"with featurize.py --energy-column;\n"
            f"  * for extxyz, write a per-atom `energies` array;\n"
            f"  * or re-featurise with --allow-missing-energy AND train with "
            f"--lambda-pred 0, which turns the property head off instead of "
            f"fitting it to fabricated values. Say so in the SI.")
    part = [j for j in range(T.shape[1]) if not finite[:, j].all()]
    if part:
        n = int((~finite).any(axis=1).sum())
        if verbose:
            print(f"  masking {n} rows ({100*n/len(T):.3f}%) with a non-finite "
                  f"target from the target scaler")
        T = T[finite.all(axis=1)]
    out["targets"] = (T.mean(0), T.std(0))
    if verbose:
        m, s = out["targets"]
        print(f"  target scaler: mean {np.round(m, 5)}  sd {np.round(s, 5)}")
    return out


def cyclical_beta(iteration, beta_max=1e-3, interval=10000):
    """The published schedule: ramp over the first half, hold over the second.

    Verified against the training log -- the logged beta*KLD term is a sawtooth
    with a 10,000-iteration period whose phase explains 93.6% of its variance.
    """
    if interval <= 0:
        return beta_max
    i = iteration % interval
    half = interval // 2
    return beta_max if i > half else (i / half) * beta_max


def describe():
    lines = ["Published VAE configuration", "=" * 40]
    for k in ("dim_z", "lr", "batch_size", "epochs", "seed", "dropout",
              "lambda1", "lambda2", "lambda3", "beta_max", "beta_schedule",
              "beta_interval", "kld_reduction", "mse_reduction"):
        lines.append(f"  {k:<16}{CONFIG[k]}")
    lines.append("")
    lines.append("Architecture")
    for k in ("encoder1", "encoder2", "latent", "decoder_hidden",
              "predictor_hidden", "decoder_layers", "latent_layers"):
        lines.append(f"  {k:<16}{CONFIG[k]}")
    lines.append(KNOWN_ISSUES)
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    tr, va, te = split_published(18058040)
    print("split_published(18,058,040):")
    print(f"  train {len(tr):,}   valid {len(va):,}   test {len(te):,}")
