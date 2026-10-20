#!/usr/bin/env python3
"""
STAGE 2 (Supplementary Section 9) -- train a VAE on a new system, at the
PUBLISHED hyperparameters.

    python train_new_system.py --features feats/si --system silicon-sw \
        --out runs/silicon-sw

The point of Supplementary Section 9 is that the FRAMEWORK generalises, not
that the water codebook does. So every system gets its own VAE, trained from
scratch, and the only thing held fixed is the procedure. That means the
hyperparameters below are not defaults to be tuned per system -- they are the
published values, and changing any of them makes the comparison meaningless:

    dim(z1) = dim(z2) = 20      dim(Z) = 40
    learning rate 1e-4          mini-batch 1000
    tanh activations            dropout 0.10
    200 epochs (fixed)          seed 123456
    lambda1 = lambda2 = 1       lambda3 = 0.1
    beta  cyclical, max 1e-3 over a 10,000-iteration period

TWO THINGS THE PUBLISHED RUN DID THAT ARE WORTH KNOWING
-------------------------------------------------------
1. beta is CYCLICAL, ramping to a maximum of 1e-3 over a 10,000-iteration
   period -- linear over the first half of each cycle, held at the maximum over
   the second. Verified against the training log: the logged beta*KLD term is a
   sawtooth whose phase explains 93.6% of its variance. An earlier version of
   this file claimed a constant beta = 1/1024; that was a misreading of a script
   that was not the one that ran, and it is withdrawn.

2. Early stopping never fired. The criterion is evaluated and logged but never
   breaks the loop, so all runs went the full 200 epochs. Reproduced here, with
   the log message, so the behaviour matches.

SPLITTING
---------
Split by FRAME, never by row. Atoms within a frame are strongly correlated, so
a random row split leaks neighbours of training atoms into validation and the
validation loss becomes optimistic. `--split frame` is the default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# NOTE: the original carried a three-directories-up sys.path.insert so
# it could find the library when run from inside the source tree. It is
# removed here: the package is installed, and leaving it in would let
# whatever sits three directories above shadow the installed icemotifs.


from icemotifs import columns as C            # noqa: E402
from icemotifs import systems as S            # noqa: E402
from icemotifs import published as P          # noqa: E402

# Transcribed from the original scripts and cross-checked against the released
# checkpoint, the released scalers and the training log. See icemotifs/published.py.
PUBLISHED = dict(P.CONFIG)
PUBLISHED["beta"] = PUBLISHED["beta_max"]


def _block_shapes(path, files):
    """(rows, ncols, dtypes) from the .npy HEADERS, one descriptor at a time.

    The obvious `[np.load(f, mmap_mode="r") for f in files]` holds one open
    memory-map per block, so at 100,125 blocks it dies with "Too many open
    files" (errno 24) before reading any data -- which is exactly how the
    Gartner stage-2 job failed on 12 Sep. The shape lives in a 128-byte header
    that needs no mapping, so read it and close the file.
    """
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


def load_blocks(path):
    """Load a directory of per-frame .npy blocks, or a single .npy matrix."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".npy"))
        if not files:
            raise SystemExit(f"no .npy blocks in {path}")
        # headers only, then preallocate -- see icemotifs/data.py for why the
        # list-then-vstack form is what ran a 64 GB job out of memory, and
        # _block_shapes for why the headers are not read through mmap.
        rows, ncol, dtypes = _block_shapes(path, files)
        n_per = sorted(set(rows))
        if len(n_per) != 1:
            print(f"  note: frames have differing atom counts {n_per}; "
                  "frame-wise splitting still works, row indexing does not")
        frame_id = np.concatenate(
            [np.full(n, i, dtype=np.int32) for i, n in enumerate(rows)])
        out_dtype = np.float32 if np.float16 in dtypes else np.float32
        total = int(sum(rows))
        print(f"  loading {len(files)} blocks: ({total:,}, {ncol}) = "
              f"{total * ncol * 4 / 2**30:.2f} GB")
        X = np.empty((total, ncol), dtype=out_dtype)
        at = 0
        for f, n in zip(files, rows):
            b = np.load(os.path.join(path, f))
            X[at:at + n] = b
            at += n
            del b
        return X, frame_id, files
    X = np.load(path)
    return X, np.zeros(len(X), dtype=np.int32), [os.path.basename(path)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True,
                   help="directory of per-frame .npy blocks, or one .npy")
    p.add_argument("--system", required=True, choices=sorted(S.REGISTRY))
    p.add_argument("--out", required=True)
    p.add_argument("--train-frac", type=float, default=0.133)
    p.add_argument("--val-frac", type=float, default=0.067)
    p.add_argument("--beta-schedule", default="cyclical",
                   choices=["cyclical", "constant"],
                   help="cyclical is what the published run used")
    p.add_argument("--epochs", type=int, default=PUBLISHED["epochs"])
    p.add_argument("--device", default=None)
    p.add_argument("--split-mode", default="block",
                   choices=["block", "published"],
                   help="block = frame-disjoint (recommended); published = the "
                        "stride-of-5 split used in the paper, which is NOT "
                        "frame-disjoint")
    p.add_argument("--mask-sentinel", type=float, default=1000.0,
                   help="target value to exclude from the target-scaler fit; "
                        "pass 0 to reproduce the original (unmasked) behaviour")
    p.add_argument("--dim-z", type=int, default=None,
                   help="latent dimension (published 40). Only the bottleneck "
                        "layer changes size, so the cost per epoch is "
                        "essentially independent of this.")
    p.add_argument("--beta-max", type=float, default=None,
                   help="weight of the KLD term (published beta = 1e-3, "
                        "cyclically annealed). This is the beta of "
                        "L = ... + beta*KLD. It is NOT the `--BETA` of the "
                        "original code, which set lambda3 = 1/BETA and is "
                        "exposed here as --lambda-pred; the two are "
                        "independent in this implementation. Raising it much "
                        "above the published value drives the posterior "
                        "towards the prior and collapses the latent.")
    p.add_argument("--lambda-pred", type=float, default=None,
                   help="weight of the property-prediction loss (published "
                        "lambda3 = 0.1). Pass 0 to switch the property head "
                        "OFF, which is the only honest setting for a dataset "
                        "whose target column is not data -- e.g. frames with "
                        "no per-atom energy, featurised with "
                        "--allow-missing-energy.")
    # ADDED 13 Sep 2026. run_pipeline.sh:326 has always passed `--seed "$SEED"`
    # when SEED is set, and submit_stages.sh:52 has always forwarded SEED -- but
    # this script never accepted the option, so every seeded run died instantly
    # with "unrecognized arguments: --seed N" (jobs 690457, 690461). The whole
    # SEED path was therefore dead on arrival, which is why the replicate runs
    # the SI's "three seeds each" sentence needs had never been produced.
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (published 123456). Changing it gives a "
                        "genuine training replicate: same data, same "
                        "hyperparameters, different initialisation and shuffle.")
    p.add_argument("--dry-run", action="store_true",
                   help="report the split and the configuration, do not train")
    a = p.parse_args(argv)

    sysdef = S.get(a.system)
    X, frame_id, files = load_blocks(a.features)
    layout = C.BY_NCOLS.get(X.shape[1])
    if layout is None:
        raise SystemExit(
            f"{X.shape[1]} columns is not a layout this pipeline knows "
            f"({sorted(C.BY_NCOLS)}). Featurisation and training disagree.")

    print(f"system   {sysdef.name}")
    print(f"features {X.shape}  ({layout.name})")
    print(f"frames   {len(np.unique(frame_id))}")

    # One seed drives the WHOLE run: the frame split here and the trainer's own
    # initialisation via cfg["seed"] below. Using PUBLISHED["seed"] here while
    # --seed changed only the trainer would give a half-replicate -- same split,
    # different initialisation -- which is not what "a different seed" means.
    run_seed = a.seed if a.seed is not None else PUBLISHED["seed"]
    rng = np.random.default_rng(run_seed)
    frames = np.unique(frame_id)
    rng.shuffle(frames)
    n_tr = max(1, int(round(a.train_frac * len(frames))))
    n_va = max(1, int(round(a.val_frac * len(frames))))
    tr, va = frames[:n_tr], frames[n_tr:n_tr + n_va]
    te = frames[n_tr + n_va:]
    print(f"split    train {len(tr)} frames / val {len(va)} / held-out {len(te)}"
          f"   (frame-wise, seed {run_seed})")

    # Record the cutoffs the FEATURES were built at, not the registry
    # defaults. Si-nucleation ran at --d-nn 2.415 (pinned in datasets.conf and
    # confirmed by the probe in every window log) while the registry says
    # 2.35, so system_config.json documented 2.35 / SOAP 5.11 for features
    # that are actually 2.415 / 5.25. The features were right; the file
    # describing them was not.
    d_nn_used, soap_rcut_used = sysdef.d_nn, sysdef.soap_rcut
    stein_used = sysdef.stein_cutoffs
    d_nn_source = "registry default (icemotifs/systems.py)"
    _meta = os.path.join(a.features, "_meta", "d_nn.json")
    if not os.path.isdir(a.features):
        _meta = None
    if _meta and os.path.exists(_meta):
        _m = _meta
    else:
        _m = None
        base = a.features if os.path.isdir(a.features) else ""
        # the pool is symlinks; follow one back to its trajectory folder
        if base:
            import glob as _glob
            for cand in sorted(_glob.glob(os.path.join(
                    os.path.dirname(base.rstrip("/")), "..", "features",
                    "*", "_meta", "d_nn.json"))):
                _m = cand
                break
    if _m:
        try:
            with open(_m) as fh:
                rec = json.load(fh)
            if isinstance(rec, dict) and "d_nn" in rec:
                d_nn_used = float(rec["d_nn"])
                sysdef2 = sysdef.__class__(**{**sysdef.__dict__,
                                              "d_nn": d_nn_used})
                soap_rcut_used = sysdef2.soap_rcut
                stein_used = sysdef2.stein_cutoffs
                d_nn_source = _m
        except Exception as exc:                                # noqa: BLE001
            print(f"  note: could not read {_m}: {exc}")
    print(f"  cutoffs recorded: d_nn {d_nn_used} A  SOAP r_cut "
          f"{soap_rcut_used} A   (from {d_nn_source})")

    cfg = dict(PUBLISHED)
    cfg.update(system=sysdef.name, epochs=a.epochs,
               beta_schedule=a.beta_schedule, layout=layout.name,
               n_features=int(X.shape[1]),
               train_frames=[int(x) for x in tr],
               val_frames=[int(x) for x in va],
               d_nn=d_nn_used, soap_rcut=soap_rcut_used,
               d_nn_source=d_nn_source,
               stein_cutoffs=[float(c) for c in stein_used])

    os.makedirs(a.out, exist_ok=True)
    if a.dim_z is not None:
        cfg["dim_z"] = a.dim_z
    if a.seed is not None:
        # cfg["seed"] feeds both the frame split above and the argv2 handed to
        # the training loop, so one value drives the whole run.
        cfg["seed"] = a.seed
        print(f"  seed {a.seed} (published 123456): this is a training "
              f"replicate, not a new configuration.")
    if a.beta_max is not None:
        cfg["beta"] = a.beta_max
        print(f"  beta (KLD weight) = {a.beta_max} "
              f"(published {PUBLISHED['beta_max']})")
    if a.lambda_pred is not None:
        cfg["lambda3"] = a.lambda_pred
        if a.lambda_pred == 0:
            print("  lambda3 = 0: the property head is OFF. No energy or "
                  "Voronoi prediction loss is optimised, and none should be "
                  "reported for this run.")
    # NOT config.json: the training loop (_fit, below) writes its own
    # config.json into the same directory and would clobber this one.
    with open(os.path.join(a.out, "system_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nconfiguration -> {a.out}/system_config.json")
    for k in ("dim_z", "lr", "batch_size", "dropout", "epochs",
              "lambda3", "beta", "seed"):
        print(f"    {k:<12} {cfg[k]}")

    if a.dry_run:
        print("\n--dry-run: nothing trained.")
        return 0

    # The scaler is fitted on the TRAINING frames only.
    from icemotifs import scalers

    mask_tr = np.isin(frame_id, tr)
    # Fitted on the TRAINING rows only, and with the sentinel targets masked --
    # both differ from the original, deliberately. See icemotifs/published.py,
    # KNOWN_ISSUES 2 and 5.
    # allow_dead_targets only when the property head is OFF. With it ON an
    # all-non-finite target is still a hard stop -- that guard is what catches
    # a NaN energy column before it trains a head against nothing. With
    # lambda3 = 0 nothing regresses the column, so refusing to fit a scaler
    # for it refuses the very remedy the error message recommends, which is
    # what blocked the Gartner run on 13 Sep.
    P.fit_scalers(X[mask_tr], layout,
                  sentinel=(a.mask_sentinel or None),
                  allow_dead_targets=(cfg["lambda3"] == 0))
    scal = scalers.fit_blocks(X[mask_tr], layout)
    scalers.save_blocks(scal, a.out)
    print(f"scaler fitted on {mask_tr.sum():,} training rows only")

    argv2 = ["--features", a.features, "--outdir", a.out,
             "--dim-z", str(cfg["dim_z"]), "--epochs", str(cfg["epochs"]),
             "--batch-size", str(cfg["batch_size"]), "--lr", str(cfg["lr"]),
             "--dropout", str(cfg["dropout"]),
             "--lambda-pred", str(cfg["lambda3"]),
             "--seed", str(cfg["seed"])]
    # The library splits by frame using a fixed atoms-per-frame count. That is
    # only meaningful when every frame has the same number of atoms, which is
    # true for a fixed-N MD run but not for anything with a varying particle
    # count -- so it is checked rather than assumed.
    per_frame = sorted({int(n) for n in np.bincount(frame_id) if n})
    if len(per_frame) == 1:
        argv2 += ["--split", "frame", "--atoms-per-frame", str(per_frame[0])]
    else:
        print(f"  frames have differing atom counts {per_frame}; falling back "
              f"to --split block so no frame is straddled")
        argv2 += ["--split", "block"]
    argv2 += ["--beta-max", str(cfg["beta"]),
              "--beta-interval", str(cfg["beta_interval"] if a.beta_schedule == "cyclical" else 0)]
    if a.device:
        argv2 += ["--device", a.device]
    return _fit(argv2)


def _fit(argv):
    """The training loop itself, moved here from `icemotifs/train.py`.

    `main()` above resolves the published hyperparameters, records the cutoffs
    and fits the scalers; this runs the epochs. The argv it is handed is the one
    `main()` builds, so the defaults below are what an unset option means --
    `--kld-reduction mean` is the code that produced the published run, and
    `--patience 50` is evaluated but, as the module docstring says, never fired.

    The heavy imports stay inside the function: `--dry-run` returns before this
    is called, which is why a dry run (and most of the test suite) never loads
    torch.
    """
    import time

    import torch

    from icemotifs import data as D
    from icemotifs import model as M
    from icemotifs import scalers as SC
    from icemotifs import train as T
    p = argparse.ArgumentParser(description="train the multimodal VAE")
    p.add_argument("--features", required=True, help=".npy feature matrix")
    p.add_argument("--outdir", required=True)
    p.add_argument("--dim-z", type=int, default=40)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-pred", type=float, default=0.1, help="l3 (paper: 0.1)")
    p.add_argument("--beta-max", type=float, default=1e-3)
    p.add_argument("--beta-interval", type=int, default=10000)
    p.add_argument("--kld-reduction", default="mean", choices=["mean", "sum"])
    p.add_argument("--split", default="frame", choices=["frame", "block", "index"])
    p.add_argument("--atoms-per-frame", type=int, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    X, layout = D.load_features(a.features)
    X = D.clean(X)
    if a.split == "frame" and not a.atoms_per_frame:
        p.error("--split frame needs --atoms-per-frame (e.g. 128000)")
    tr, va, te = D.make_split(len(X), a.split, a.atoms_per_frame, seed=a.seed)
    D.split_report(a.split, tr, va, te, len(X), a.atoms_per_frame)

    # scalers are fitted on TRAIN ONLY. The original fitted them on the whole matrix
    # before splitting, which leaks the standardisation statistics into the test set.
    scal = SC.fit_blocks(X[tr], layout)
    SC.save_blocks(scal, a.outdir)
    for s in scal:
        s.to(a.device)

    model = M.build(scal[0], scal[1], dim_z=a.dim_z,
                    regression=len(layout.targets), dropout=a.dropout).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    tl = D.loader(X[tr], a.batch_size, shuffle=True)
    vl = D.loader(X[va], a.batch_size, shuffle=False)

    print(f"  dim(Z)={a.dim_z}  targets={len(layout.targets)}  device={a.device}")
    hist, best, bad, it = [], float("inf"), 0, 0
    t0 = time.time()
    for ep in range(a.epochs):
        trm, it = T.run_epoch(model, tl, layout, scal, opt, lambdas=(1., 1., a.lambda_pred),
                            beta_max=a.beta_max, beta_interval=a.beta_interval,
                            kld_reduction=a.kld_reduction, it0=it, device=a.device)
        vam, _ = T.run_epoch(model, vl, layout, scal, None, lambdas=(1., 1., a.lambda_pred),
                           beta_max=a.beta_max, beta_interval=a.beta_interval,
                           kld_reduction=a.kld_reduction, it0=it, device=a.device)
        hist.append({"epoch": ep + 1, **{f"train_{k}": v for k, v in trm.items()},
                     **{f"val_{k}": v for k, v in vam.items()}})
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep+1:4d}  train {trm['loss']:.5f}  val {vam['loss']:.5f}"
                  f"  (stein {vam['stein']:.5f}  soap {vam['soap']:.5f}  pred {vam['pred']:.5f})",
                  flush=True)
        if vam["loss"] < best - 1e-4:
            best, bad = vam["loss"], 0
            M.save(model, os.path.join(a.outdir, "model.pt"),
                   extra={"epoch": ep + 1, "val_loss": best, "layout": layout.name})
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep+1}: no improvement for {a.patience} epochs")
                break

    # unlike the original, "best" and "last" are both written and clearly named
    M.save(model, os.path.join(a.outdir, "model_last.pt"), extra={"epoch": len(hist)})
    with open(os.path.join(a.outdir, "history.json"), "w") as fh:
        json.dump(hist, fh, indent=1)
    with open(os.path.join(a.outdir, "config.json"), "w") as fh:
        json.dump(vars(a), fh, indent=1)
    print(f"  done in {time.time()-t0:.1f}s -> {a.outdir}/model.pt (best val {best:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
