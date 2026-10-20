# Examples: seven published analyses

Each folder is one dataset from the paper: the seven-stage pipeline that produced
it as both a notebook and scripts, the published figures to compare against, and
ten frames of the trajectory so it runs immediately.

## Quick start (seven minutes, measured)

```bash
pip install -e ..
cd ldl-hdl                 # the cheapest folder: 192 atoms/frame
export STRIDE=1            # the sample is ALREADY strided
python scripts/01_featurize.py sample-data/192-235.0-3200-hi-water.sample.extxyz
python scripts/02_train_vae.py && python scripts/03_project.py
python scripts/04_cluster.py  && python scripts/05_relabel.py
python scripts/06_analyze.py  && python scripts/07_figures.py
```

Everything lands in `./run/`, or `$OUT` if you set it. `analysis.ipynb` runs the
same seven steps with a note on each.

- **`STRIDE=1` is not optional on the samples.** The ten frames are already
  spread across the published trajectory; applying the published stride again
  keeps 1 of 10 at `ldl-hdl`. The notebook sets it whenever `TRAJECTORY` points
  into `sample-data/`.
- **Ten frames prove the pipeline runs.** The parameters in `scripts/config.py` are the published ones,
  tuned for thousands of frames.
- **Cost per frame**, one core: `ldl-hdl` 13.8 s (192 O of 576 atoms), `silicon`
  **325 s** (32,768 atoms) — about 55 minutes for its sample. Start with
  `ldl-hdl`.
- For real numbers, use a full trajectory: yours, or the deposit
  (<https://doi.org/10.5281/zenodo.20483266>).

## What is in a folder

| | |
|---|---|
| `analysis.ipynb` | the whole pipeline as a notebook |
| `sample-data/` | 10 frames of the published trajectory, gzipped; the readers open `.gz` directly |
| `scripts/` | the same seven steps as scripts, plus `config.py` |
| `reference/` | figures and a movie from the PUBLISHED analysis |
| `reproduce-figures/` | three of those sheets, REDRAWN by the notebook |

---

## The seven steps

| # | script | what it does | the choice inside it |
|---|---|---|---|
| 1 | `01_featurize.py` | Steinhardt + SOAP per atom | measures `d_nn` and stops if it disagrees with the registry |
| 2 | `02_train_vae.py` | trains the VAE on those descriptors | `lambda_pred`, the `beta` schedule, `dim_z` |
| 3 | `03_project.py` | encodes to `Z`, then PCA/UMAP to 2-D | the UMAP seed, and `n_jobs` (it changes results) |
| 4 | `04_cluster.py` | HDBSCAN clusters, then k-means over the intermediate regions | `min_cluster_size`, `n_kmeans` |
| 5 | `05_relabel.py` | ranks motifs along one projection axis | **which axis** — see below |
| 6 | `06_analyze.py` | per-motif table and populations over time | `min_percent` |
| 7 | `07_figures.py` | the figure sheets | the colour key is fixed by step 5 |


## The seven folders

| folder | system | kind | stride | axis | what it tests |
|---|---|---|---|---|---|
| `nucleation-slow/` | `water-mlbop` | filelist | 5 | 0 | the same transition at a 2.5× slower quench |
| `heterogeneous-1/` … `-4/` | `water-mlbop` | dump | 2 | 1 | four interfaces; which ice motif each selects |
| `silicon/` | `silicon-sw` | dump | 1 | 1 | a different element and potential |
| `ldl-hdl/` | `water-mw` | extxyz | 500 | 0 | a transition with no crystal at either end |

Numbered as the polymorphs are in the paper: `-1/` is `8154387`, `-2/` is
`8255250`, `-3/` is `8255188` (block prefix `822188`), `-4/` is `8157657`.

## What the published run found

| folder | HDBSCAN clusters | intermediate fraction |
|---|---|---|
| `nucleation-slow/` | 8 | 29.3 % |
| `heterogeneous-1/` … `-4/` | 20 | 12.5 % |
| `silicon/` | 12 | 36.3 % |
| `ldl-hdl/` | 7 | 21.4 % |

---

## Reproducing the published figures

```bash
python tools/run-example-notebooks.py --runs-root /path/to/the/runs/tree
```

- `reference/` is what the paper published; `reproduce-figures/` is what this
  code redraws from the published run's own arrays. 
- Each notebook ends with a **reproduction section** for three main figures:
  `05-umap-by-properties`, `dist-merged.major`, `09-composition-ring-density`.

- **The movies and the figures should use the same colormap.** Both colour
  `motif_relabel`; Use `jet` colormap for the four `heterogeneous-*/`; `ldl-hdl/` and `silicon/`. 

## Adapting a folder to your own data

Copy the closest folder, point `TRAJECTORY` (notebook) or argv (scripts) at your
trajectory, set `SYSTEM` in `scripts/config.py`. Three things catch people:

1. **The UMAP project is not a default.** Which projection axis carries the structure
   differs between datasets:
   ```bash
   icemotifs relabel --embedding run/proj/embedding.npy --labels run/motifs/labels.npy \
                     --out run/motifs --axis-from your_property.npy
   ```
   It picks by correlation and prints both axes.
2. **The heterogeneous folders cluster separately**, so motif 3 in `-1/` and
   motif 3 in `-2/` are unrelated. The published comparison pooled all four
   through one clustering and one relabel; to reproduce it, featurise the four
   into one directory and run steps 2–5 once.
