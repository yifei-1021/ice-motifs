#!/usr/bin/env python3
"""
STAGE 3 (Supplementary Section 9) -- encode a featurised trajectory to Z.

Two models can do the encoding, and the difference matters:

  A. A VAE TRAINED ON THIS SYSTEM, by stage 2.

        python encode_new_system.py --features feats/si --run runs/silicon-sw \
            --out runs/silicon-sw/Z.npy

  B. THE PUBLISHED VAE, with no retraining at all -- the released
     `model.last.pth` and its three scalers, applied to new trajectories.

        python encode_new_system.py --features mW-nucleation/runs/_pool \
            --published-model  main-dataset-and-model/model.last.pth \
            --published-scalers main-dataset-and-model \
            --system water-mw \
            --out mW-nucleation/runs-published/z.npy

     This is the transfer question: does the motif map the paper already
     published describe trajectories it never saw? Nothing is fitted here, so
     the answer is not a fit. The published latent space, its scalers and its
     cluster vocabulary all stay fixed and the new frames are simply placed in
     them, which means the resulting Z is directly comparable to the published
     Z -- unlike a freshly trained VAE, whose latent axes are its own.

BEFORE IT ENCODES WITH THE PUBLISHED MODEL IT CHECKS TWO THINGS
---------------------------------------------------------------
1. The CUTOFFS. The checkpoint was trained at SOAP r_cut 6.0 A and Steinhardt
   3.2/4.8/6.0 A. `water-mw` and `water-mlbop` reproduce those exactly, so the
   published weights read their features directly. `silicon-sw` does not: its
   cutoffs are the same reduced multiples of a different d_nn, so column 40
   means a different radius. That mismatch is completely silent -- 281 columns
   either way -- and it is worth 1943x in SOAP reconstruction error. Pass
   --allow-cutoff-mismatch to do it anyway, and label the result a transfer
   experiment when you write it up.

2. The DOMAIN. Standardised with the published scalers, where do the new rows
   sit? Centred means interpolation; many standard deviations out means the
   encoder is extrapolating and the latent coordinates are an extension of the
   published map rather than a position in it.

The scalers are never re-fitted here, in either mode. Re-fitting at inference
is the classic way to make a model look better than it is: the statistics would
then include the frames you are trying to hold out.
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


from icemotifs import columns as C   # noqa: E402,F401


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="stage-2 output directory (model.pt + scaler_*.npz)")
    g.add_argument("--published-model",
                   help="the released full-pickle checkpoint, model.last.pth")
    p.add_argument("--published-scalers", metavar="DIR",
                   help="folder holding steinsc.pkl, soapsca.pkl and regrsca.pkl. "
                        "Defaults to the folder --published-model is in.")
    p.add_argument("--system", help="system name, for the cutoff check "
                                    "(required with --published-model)")
    p.add_argument("--allow-cutoff-mismatch", action="store_true",
                   help="encode even though the features were built at "
                        "different cutoffs than the checkpoint was trained at")
    p.add_argument("--no-domain-check", action="store_true")
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    from icemotifs import data as D, encode as E, model as M, scalers as SC

    device = a.device or "cpu"
    provenance = {}

    # ---------------------------------------------------------- pick the model
    if a.run:
        cfg_path = os.path.join(a.run, "system_config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        print(f"run      {a.run}")
        if cfg:
            print(f"  system {cfg.get('system')}  dim(Z) {cfg.get('dim_z')}  "
                  f"beta {cfg.get('beta')} ({cfg.get('beta_schedule')})")
        scal = SC.load_blocks(a.run)
        model = M.load(os.path.join(a.run, "model.pt"), scal[0], scal[1],
                       map_location=device)
        provenance = {"mode": "trained-here", "run": os.path.abspath(a.run)}
    else:
        from icemotifs import compat, published, systems
        if not a.system:
            raise SystemExit("--published-model needs --system (water-mw, "
                             "water-mlbop, silicon-sw): the cutoff check "
                             "cannot be skipped by leaving it out")
        sysdef = systems.get(a.system)
        print(f"model    {a.published_model}   (PUBLISHED -- nothing is trained)")

        ok, lines = published.cutoff_report(sysdef)
        print("\ninput space:")
        print("\n".join(lines))
        if not ok and not a.allow_cutoff_mismatch:
            raise SystemExit(
                "\nrefusing to encode. Re-run with --allow-cutoff-mismatch if a "
                "transfer\nexperiment is what you want, and say so in the text.")

        print()
        model = compat.load_legacy(a.published_model, map_location=device)
        sdir = a.published_scalers or os.path.dirname(
            os.path.abspath(a.published_model))
        scal = compat.load_published_scalers(sdir, model=model)
        model.scaler1, model.scaler2 = scal[0], scal[1]
        provenance = {"mode": "published", "system": a.system,
                      "model": os.path.abspath(a.published_model),
                      "scalers": os.path.abspath(sdir),
                      "cutoffs_match_published": bool(ok)}

    # ---------------------------------------------------------------- features
    print()
    X, layout = D.load_features(a.features)
    X = D.clean(X)

    if not a.run and not a.no_domain_check:
        from icemotifs import published
        worst, lines = published.domain_report(X, layout, scal)
        print("\nwhere the new rows sit in the published feature distribution:")
        print("\n".join(lines))
        provenance["domain_max_shift_sd"] = round(float(worst), 3)

    # ----------------------------------------------------------------- encode
    print()
    # want_recon=False: the reconstruction is another (n_rows, 281) array and
    # this script does not write it. For the 19.2M-row sets that is 21.6 GB
    # held, plus as much again while it is stacked.
    Z, MU, LV, _, metrics = E.encode_array(
        model, X, layout, scal, a.batch_size, device, deterministic=True,
        want_recon=False)
    logsig = E.log_sigma_norm(LV)
    metrics.update(provenance)

    outdir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    # deterministic=True above, so Z is the latent MEAN, not a sample.
    np.save(a.out, np.asarray(Z, dtype=np.float32))
    base = os.path.splitext(a.out)[0]
    np.save(base + "_logsigma.npy", np.asarray(logsig, dtype=np.float32))
    with open(os.path.join(outdir, "encode_metrics.json"), "w") as fh:
        json.dump({k: (v if isinstance(v, (str, bool, int, dict)) else float(v))
                   for k, v in metrics.items() if v is not None}, fh, indent=2)

    print(f"Z             {np.shape(Z)}      -> {a.out}")
    print(f"||log sigma|| {np.shape(logsig)}  -> {base}_logsigma.npy")
    print(f"  mean {float(np.mean(logsig)):.4f}  sd {float(np.std(logsig)):.4f}")
    print(f"  MSE  steinhardt {metrics['mse_steinhardt']:.6f}   "
          f"soap {metrics['mse_soap']:.6f}")
    if metrics.get("mse_per_target"):
        print("       " + "   ".join(f"{k} {v:.6f}"
                                     for k, v in metrics["mse_per_target"].items()))
    if not a.run:
        # The numbers the released checkpoint scores on its OWN test rows, so
        # the line above can be read rather than merely recorded. Measured on
        # 50,134 rows of the published trajectory, 9 Sep 2026.
        print("\n  published model on its own data:  steinhardt 0.003801   "
              "soap 0.012566")
        print("  a SOAP figure far above ~0.013 means these features are not "
              "in the\n  space the checkpoint was trained on, whatever the "
              "cutoff check said.")
    print("\nnext: python ../4-projection/pca_umap.py "
          f"--z {a.out} --out {os.path.dirname(a.out) or '.'}/proj")
    return 0


if __name__ == "__main__":
    sys.exit(main())
