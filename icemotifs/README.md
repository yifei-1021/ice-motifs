# The library, file by file

The modules **are** the procedure. The scripts in `examples/*/scripts/`, the
notebooks and the `icemotifs` command are front ends to them — so whatever you
want to change has exactly one place to change it.

## Where do I change …?

| I want to … | edit |
|---|---|
| add a system, or fix a `d_nn` | `systems.py` — one dataclass entry |
| change a descriptor or a cutoff | `systems.py` (cutoffs are multiples of `d_nn`), then `descriptors.py` |
| change the VAE, or the property head's weight | `model.py`, `train.py` (`lambda_pred` lives in the loss) |
| change the projection or its seed | `pipeline/project.py` |
| change the clustering | `cluster.py` — the published two-stage recipe, in one place |
| change what a figure looks like | `figures.py` |
| add a column to the feature matrix | `columns.py` **first** — it is the single source of the layout |

## The stages

`pipeline/` holds one module per stage, each with a `main(argv)` that `cli.py`
dispatches to, and **they are the only command-line entry points** — everything
below is imported, not run. `icemotifs all` runs 1–5, leaving 6 and 7 unchained
because `relabel` needs an axis and the informative one differs between
systems.

| | file | what it does | writes |
|---|---|---|---|
| 0 | `pipeline/check.py` | measures `d_nn` from frame 0, prints it beside the registry value | nothing |
| 1 | `pipeline/featurize.py` | reads dump / extxyz / filelist through one path, checks `d_nn`, shards output | `features/` |
| 2 | `pipeline/train_vae.py` | trains the VAE at the published hyperparameters, cyclical β schedule | `vae/` |
| 3 | `pipeline/encode_run.py` | runs the trained encoder over the feature matrix | `Z.npy`, `Z_logsigma.npy` |
| 4 | `pipeline/project.py` | PCA → UMAP, saving the fitted reducer so it is reused, not refitted | `proj/` |
| 5 | `pipeline/cluster_motifs.py` | HDBSCAN clusters, then k-means over the intermediate regions they left | `motifs/` |
| 6 | `pipeline/relabel_motifs.py` | ranks motifs along one projection axis, writes the colour legend | `relabel.csv/.json` |
| 7 | `pipeline/write_frames.py` | per-atom labelled frames for a renderer; refuses a count mismatch | `frames/*.extxyz` |

## The modules they are built from

| file | what it holds |
|---|---|
| `systems.py` | the registry — one dataclass per system; every cutoff derived from its `d_nn` |
| `lammps.py` | the LAMMPS dump reader; sorts every frame by atom id, because dump order is not stable |
| `descriptors.py` | Steinhardt + SOAP + Voronoi, the 281 columns |
| `columns.py` | the column layout, declared once, so no stage can disagree about where a block starts |
| `data.py` | datasets and the train/validation split |
| `scalers.py` | per-block standardisation, fitted on training rows only, stored with the model |
| `model.py` | the multimodal VAE: encoder, decoder, and the property head weighted by `lambda_pred` |
| `train.py` | the β schedule, the KLD, and one epoch's forward/backward — the arithmetic stage 2 calls |
| `encode.py` | the trained encoder applied to a feature matrix |
| `cluster.py` | the published two-stage clustering, in one place |
| `relabel.py` | the rank along a projection axis — `motif` identifies, `motif_relabel` colours |
| `motifs.py` | level-2 analysis: motif transitions and lifetimes |
| `rings.py` | ring statistics — three definitions, and why the choice changes the answer |
| `analysis.py` | per-motif tables: population, mean properties, ring composition |
| `figures.py` | motif map, property panels, populations, distributions, colour key |
| `dip.py` | Hartigan's dip statistic with a bootstrap p-value, for testing bimodality |
| `published.py` | the published configuration, transcribed from the original scripts and verified against them |
| `compat.py` | loads legacy checkpoints — `torch` 2.6 flipped the `weights_only` default |
| `cli.py` | the `icemotifs` command |

## Three invariants worth not breaking

- **Every cutoff is a multiple of `d_nn`.** Nothing hardcodes an ångström. A
  wrong `d_nn` rescales all of them together and nothing crashes.
- **One column layout.** `columns.py` is the only place that knows where the
  Steinhardt block ends and SOAP begins; `scalers.py` standardises per block.
- **The reducer is saved with the fit/transform split that produced it.** Stage
  4 fits on a subsample and transforms the rest through that same reducer, and
  `projection.json` records which rows were which. Applying a stored reducer to
  a *different* trajectory is deliberately not wired into the stage: load
  `proj/reducer.pkl`, call `transform` yourself, and say that you did.
