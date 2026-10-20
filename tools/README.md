# How to run this pipeline, and what each stage decides

The reference for someone reproducing an analysis: the numbering, the stages and
the choice inside each one, the per-system numbers to check first, and how to
register a trajectory of your own. **Nothing in this directory is needed to use
the package** — it holds this guide, the tests and the scripts that maintain the
repository.

- [Which numbering am I reading?](#which-numbering-am-i-reading)
- [Run the tests first](#run-the-tests-first)
- [The stages, and the choices inside them](#the-stages-and-the-choices-inside-them)
- [Per-system settings — check these before you run](#per-system-settings--check-these-before-you-run)
- [Featurising your own trajectory](#featurising-your-own-trajectory)
- [The maintenance scripts](#the-maintenance-scripts)

---

## Which numbering am I reading?

Two numberings are in use and they diverge from 3 onward. **This file, the root
`README.md` and `icemotifs/README.md` use the package stages; `examples/` and the
notebooks use the script steps.**

| package stage (`icemotifs <cmd>`) | example script | what it produces |
|---|---|---|
| 0 `check` | — | measured `d_nn`, printed, nothing written |
| 1 `featurize` | `01_featurize.py` | `features/` |
| 2 `train` | `02_train_vae.py` | `vae/` |
| 3 `encode` | `03_project.py` does both | `Z.npy` |
| 4 `project` | ⤶ | `proj/` — embedding **and the fitted reducer** |
| 5 `cluster` | `04_cluster.py` | `motifs/` |
| 6 `relabel` | `05_relabel.py` | `relabel.csv` |
| — | `06_analyze.py` | `motif_table.csv` |
| — | `07_figures.py` | the figure sheets |
| 7 `frames` | — | labelled extxyz for a renderer |

`icemotifs all` runs stages 1–5. Stages 6 and 7 are not chained on: `relabel`
needs a projection axis, and the informative axis differs between systems.

Each stage is also importable with a `main(argv)` —
`icemotifs.pipeline.featurize`, `.train_vae`, `.encode_run`, `.project`,
`.cluster_motifs`, `.relabel_motifs`, `.write_frames`, `.check`.

## Run the tests first

```bash
python -m pytest tools/            # or: python tools/test_pipeline.py
```

30 checks, about a minute, no trajectory needed — `make_synthetic.py` builds a
diamond-like lattice with noise in two "phases" and they run against that. Both
invocations run the same checks; the script form prints each name with `PASS` or
`FAIL` and exits with the failure count.

They cover what goes **silently** wrong rather than what raises: cutoff
rescaling, the reader's atom-id ordering, the column layout width, the
two-stage clustering contract, reducer persistence, and the refusal to fabricate
a missing energy column.

---

## The stages, and the choices inside them

```
trajectory ──1─► features ──2─► VAE ──3─► Z (40-D) ──4─► 2-D embedding ──5─► motifs
```

### Stage 1 — featurisation: 281 columns, every cutoff from `d_nn`

| columns | what | source |
|---|---|---|
| 0–26 | 27 Steinhardt order parameters | pyscal3, averaged (Lechner–Dellago) |
| 27–278 | 252 SOAP components | dscribe |
| 279 | Voronoi volume | pyscal3 |
| 280 | per-atom potential energy | your `energy_column` |

**The stage measures `d_nn` itself and stops if it disagrees with the registry
by more than `--dnn-tol` (3%).** It is a median over `--dnn-frames` (5) kept
frames, cached per trajectory in `d_nn.json`, and `--d-nn` outranks both. That
guard is the one worth understanding — see
[per-system settings](#per-system-settings--check-these-before-you-run).

**If your trajectory carries no per-atom energy, this stage stops too**, naming
the three ways out: dump the LAMMPS `pe/atom` compute, write a per-atom
`energies` array into the extxyz, or pass `--allow-missing-energy` and then
train with `--lambda-pred 0`. Do not take the third without the fourth: the
refusal exists because a 140-window run once wrote an all-NaN energy column,
fitted a target scaler that printed `mean [31.5031 nan]`, and trained a
regression head against nothing without stopping.

### Stage 2 — training: `lambda_pred` is the setting to understand

```
L = 1·MSE(Steinhardt) + 1·MSE(SOAP) + lambda_pred·MSE(volume, energy) + beta·KLD
```

**Columns 279 and 280 are never encoder inputs.** The encoder sees Steinhardt
and SOAP only; those two columns enter solely as targets of a regression head on
the latent, weighted by `lambda_pred`.

**The default is 0.1, the published value.** The original script set
`lambda3 = 1/beta` with `beta = 10` and named its output `TRAIN.{AL}.{BETA}`, so
the deposited `TRAIN.T8.10` decodes to `0.1`.

> With `lambda_pred > 0` the latent is **explicitly trained so that a head on
> `z` recovers per-atom Voronoi volume and energy.** If your motifs then
> separate on density or on energy, that is partly by construction — it is not
> independent evidence that a structural embedding discovered thermodynamics.

Set `lambda_pred = 0` for a latent shaped by local structure alone, at the cost
of a slightly different vocabulary. It is also what an energy-free trajectory
needs: with the head on an all-non-finite target is a hard stop at the scaler
fit, and with it off the column is neutralised to mean 0 sd 1 and **the
prediction loss must not be reported**.

Our *ab initio* LDL/HDL run is exactly that case: energy non-finite in all
19,224,000 rows, `lambda_pred = 0`. Its `voronoi_volume` MSE of 1.0097 — chance,
on a standardised target — confirms the head never trained.

#### `beta` is a different weight, and the name is a trap

`--beta-max` weights the **KLD** term — published **1e-3**, annealed cyclically
over 10,000-iteration cycles. The `beta = 10` above refers to the *property*
weight, from an original script where one `--BETA` argument set `lambda3`.

**Here the two are independent**: `--beta-max` moves only the KLD term,
`--lambda-pred` only the property head, and a test asserts neither writes the
other's config key.

- **Scale.** Stay near 1e-3. Values of order 1 are three orders of magnitude
  larger, drive the posterior to the prior and collapse the latent; a meaningful
  sweep is 1e-4, 1e-3, 1e-2.
- **Direction of the uncertainty read-out.** `log sigma` is negative in every
  latent dimension for a trained model at these settings, so a *larger*
  `||log sigma||` is a *smaller* posterior variance — more confidence, not less.
  Raising `beta` pushes sigma towards 1 and shrinks that magnitude.

On one 8.96M-atom system, sweeping `beta` over 10× and `lambda_pred` over 100×
moved the cluster count by about as much as re-seeding the training does, and
left the layout of the projection essentially unchanged. One system: a reason to
check, not a reason not to.

### Stage 4 — projection: keep the reducer

PCA to 20 components, then UMAP to 2, fitted on `--fit-subsample` rows and
applied to the rest. **The fitted reducer is written to disk**, which is what
lets you `transform` a *different* trajectory into the same plane later without
refitting — the only honest way to put two systems in one embedding.

UMAP `transform` on unseen points is an approximation, not the fitted embedding.
If you push a foreign trajectory through, report how far outside the fitted
domain it lands rather than labelling it unconditionally.

### Stage 5 — clustering: two kinds of label, and what to quote

HDBSCAN finds clusters, and what it leaves unclaimed is the **intermediate
regions** — the continuum between clusters, which has no density peak for it
to find. `n_kmeans` k-means cells tile that continuum, every remaining point
going to its nearest cell, so the vocabulary has two kinds of member:

* **HDBSCAN clusters** — regions the data actually concentrates in;
* **k-means clusters** — a chosen partition of the intermediate regions.

**Quote the HDBSCAN cluster count and the intermediate fraction.** The motif
total is the sum of a result and your own `n_kmeans`; on our runs the
intermediate regions hold 11–38 % of atoms, so the total reports a setting as
much as a system. `clustering.json` records all three, under the field names
`n_density_maxima`, `n_kmeans` and `outlier_fraction` — the keys and the printed
lines keep the older wording, and they are the same three numbers.

Labels are **0-based** in `labels.npy`, with `-1` for the HDBSCAN noise class
where one is kept; figures in the paper are 1-based. Fix the convention once in
your own analysis and assert it — mixing the two is the commonest way to get a
plot that looks right and is wrong by one motif.

---

## Per-system settings — check these before you run

Everything that differs between systems is one dataclass entry in
`icemotifs/systems.py`. Two values are yours to confirm; the rest are derived.

| system        | d_nn (Å) | SOAP r_cut | Steinhardt        | ring | source of d_nn |
|---------------|---------:|-----------:|-------------------|-----:|----------------|
| `water-mlbop` |     2.76 |       6.00 | 3.20 4.80 6.00    | 3.20 | published run  |
| `water-mw`    |     2.76 |       6.00 | 3.20 4.80 6.00    | 3.20 | same as ML-BOP |
| `silicon-sw`  |     2.35 |       5.11 | 2.72 4.09 5.11    | 2.72 | diamond-Si NN  |

`water-mlbop` reproduces the published cutoffs exactly — that is `test_systems`,
not a coincidence. Every other system is measured at the same **coordination
shells** rather than the same absolute radius, so a difference between systems
is a statement about structure and not about the cutoff.

**Measure `d_nn` on your own trajectory before trusting the table:**

```bash
icemotifs check --system water-mw   --dump mw_quench.lammpstrj
icemotifs check --system silicon-sw --dump si_quench.lammpstrj
```

`icemotifs check` reads **LAMMPS dumps only**, one frame, and warns past 5 %, so
for extxyz or filelist input rely on stage 1 instead — it measures `d_nn` for
every input kind and stops past 3 %. Either way, if the measured value really
differs, pass `--d-nn` deliberately and say which value you used.

### VAE hyperparameters — do not tune these per system

They live in `icemotifs/published.py`, transcribed from the original training
scripts and cross-checked against the released checkpoint, the released scalers
and `logger.log`.

| parameter        | value    |
|------------------|----------|
| dim(z1), dim(z2) | 20, 20   |
| dim(Z)           | 40       |
| learning rate    | 1e-4     |
| mini-batch       | 1000     |
| activation       | tanh     |
| dropout          | 0.10     |
| epochs           | 200 fixed|
| seed             | 123456   |
| λ1, λ2, λ3       | 1, 1, 0.1|
| β                | cyclical, max 1e-3, period 10,000 iterations |

Supplementary Section 9 argues the *framework* generalises, and that argument
only holds if the procedure is identical across systems. These are fixed, not
defaults.

Two of them were settled against the training log rather than the manuscript:

* **β really is cyclical** — a sawtooth with a 10,000-iteration period, ramping
  over the first half and holding at 1e-3 over the second. (An earlier note here
  claimed a constant 1/1024; that was wrong.)
* **Early stopping never fires.** The log prints "Stopping Training at Epoch 65"
  and then runs 134 more epochs to 200 — the message is emitted, the loop does
  not break. Confirmed twice in one run, at epochs 65 and 193.

---

## Featurising your own trajectory

### 1. Measure `d_nn` first

Every cutoff is a fixed multiple of the first-neighbour distance:

| quantity | multiple of `d_nn` |
|---|---|
| SOAP `r_cut` | 2.175 |
| Steinhardt shells | 1.160, 1.740, 2.175 |
| ring network | 1.160 |
| coordination | 2.175 |

A wrong `d_nn` therefore rescales the SOAP radius, all three Steinhardt shells
and the ring network *together*. Nothing crashes; the motifs simply describe a
different length scale than you think.

```bash
icemotifs check --system water-mw --dump /path/to/your.lammpstrj
```

This prints the first peak of g(r) from frame 0 beside the registry value. They
should agree to a few hundredths of an ångström.

### 2. Add the entry

```python
# in icemotifs/systems.py
MY_SYSTEM = System(
    name="my-system",
    d_nn=3.14,                   # measured in step 1, angstrom
    fmt="lammps-dump-text",      # or "extxyz"
    symbol="O",                  # element written into the ASE frames
    mass=18.015,                 # g/mol, only used for the Voronoi density
    energy_column="c_pe",        # dump column with per-atom PE, or None
    type_map={1: "O"},           # LAMMPS numeric type -> element
)
```

`icemotifs systems` will then list it.

**The three fields that catch people:**

- **`energy_column`** — the per-atom potential-energy column, `c_pe` in some of
  our runs and `c_1` in others depending on what the LAMMPS input called the
  compute; read it off the `ITEM: ATOMS` header (`id type x y z` **`c_pe`**
  `c_cn ...`). Set it to `None` if there is none, and then read the
  `lambda_pred` section above, because stage 1 stops rather than featurise a
  fabricated target.
- **`type_map`** — dumps carry numeric types, not elements, and the reader
  restricts to the species you map. A 576-atom frame yielding 192 feature rows
  is species restriction working, not a bug.
- **`fmt`** — `lammps-dump-text` or `extxyz`, the ASE format of the frames, not
  to be confused with `--kind`, which says how the *input* is packaged. For
  extxyz the atom order is file order; for dumps the reader **sorts every frame
  by id**, because row *i* of a memory-order dump is otherwise a different atom
  each frame — which is what makes the transition matrices meaningful, so do not
  bypass it.

### 3. Run

```bash
icemotifs all --system my-system --input /path/to/your.lammpstrj \
              --kind dump --out runs/my-system --stride 20
```

`--kind` is `dump`, `extxyz` or `filelist`, the last for trajectories written
one frame per file — pass an `ls -R` style listing and the stride applies to
that list. To run a stage at a time instead, see
[the stages](#the-stages-and-the-choices-inside-them).

### 4. Read the result

`runs/my-system/motifs/clustering.json`. **Report the HDBSCAN cluster count and
the intermediate fraction**, not the motif total — the total adds your chosen
`n_kmeans` tiles to a measured number, and on our runs those tiles hold 11–38 %
of the atoms.

---

## The maintenance scripts

These keep the repository consistent; they are not part of an analysis. Run them
from the repository root.

| script | what it does |
|---|---|
| `make-example-notebooks.py` | generates `examples/<dataset>/analysis.ipynb` from that folder's `01..07` scripts — **the notes come from the scripts' docstrings**, so the two front ends cannot drift apart |
| `run-example-notebooks.py` | executes each notebook's reproduction section against a published run, stores the figures, and palette-reduces them |
| `shrink-figure-png.py` | palette-reduces a stored panel at full resolution, reserving exactly the motif colours that panel draws plus every second viridis entry |
| `shrink-reference-gif.py` | reduces an OVITO render for `reference/`, or rescales an already-reduced one without rebuilding its palette |
| `make-deposit.sh` | scans the tree for machine-identifying content, then builds the deposit archive |

```bash
python tools/make-example-notebooks.py --check          # report stale, write nothing
python tools/make-example-notebooks.py                  # regenerate all seven
python tools/run-example-notebooks.py --runs-root /path/to/published/runs
python tools/shrink-figure-png.py --check               # report, write nothing
python tools/shrink-reference-gif.py render.gif out.gif 260 3 128   # first pass
python tools/shrink-reference-gif.py --rescale in.gif out.gif 185   # second pass
bash tools/make-deposit.sh
```

What each one is careful about:

- **The notebooks are generated, so an edit made in one is lost.** Change the
  script docstring, not the notebook. `--check` reports a stale notebook without
  rewriting it.
- **One global palette per movie, built from frames spanning the whole
  trajectory.** A per-frame palette shifts a motif's hue between frames; a
  palette built from the opening frames has no entry for motifs that appear
  later, and they land on whatever is nearest. The file size is the same whether
  the palette is right or wrong, so size is not the check — compare against the
  source at two timepoints by eye.
- **A palette reduction must not move a motif colour.** `shrink-figure-png.py`
  verifies every reserved colour is still exact before writing, and refuses an
  image with transparency rather than flatten it onto a guessed background.
- **`make-deposit.sh` names no machine, filesystem or user**, because it ships
  with the package; it scans for the general shapes instead.

The shipped movies were made in two passes — the first against the author's
full-size OVITO render, the second once the gallery width was settled. A fresh
render can go straight to the final width in one pass; the second pass exists so
already-reduced files could be made smaller without returning to the source, and
it reuses the colour set rather than choosing a new one.
