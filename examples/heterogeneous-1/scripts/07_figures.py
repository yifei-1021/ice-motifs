"""7. Figures, coloured by motif_relabel.

    python scripts/07_figures.py

Every panel is coloured from the SAME relabel table, over the full vocabulary
1..n. That is what makes the panels comparable with each other; colouring each
on the motifs it happens to contain would rescale the ramp per panel and the
same motif would change colour between figures.
"""
import csv, json, os, sys
import numpy as np
from icemotifs import figures as F
from icemotifs.analysis import population_over_time
from config import OUT, TITLE, DT_NS, need_stage, load_properties, n_atoms_per_frame

need_stage(f"{OUT}/motifs", "scripts/04_cluster.py")
if not os.path.exists(f"{OUT}/motifs/relabel.csv"):
    print("run 05_relabel.py first: the figures are coloured by its ranking",
          file=sys.stderr)
    sys.exit(1)

emb = np.load(f"{OUT}/proj/embedding.npy")
labels = np.load(f"{OUT}/motifs/labels.npy")
ranks = {int(r["motif"]): int(r["motif_relabel"])
         for r in csv.DictReader(open(f"{OUT}/motifs/relabel.csv"))}
n_motifs = json.load(open(f"{OUT}/motifs/clustering.json")).get("n_motifs", len(ranks))
fig_dir = f"{OUT}/figures"; os.makedirs(fig_dir, exist_ok=True)

F.motif_map(emb, labels, ranks, n_motifs, f"{fig_dir}/motif-map.png", title=TITLE)
props = load_properties(OUT, len(labels))
if props:
    F.property_panels(emb, props, f"{fig_dir}/properties.png")
    F.distributions(props, labels, ranks, n_motifs, f"{fig_dir}/distributions.png",
                    title=TITLE)
n_at = n_atoms_per_frame(OUT, len(labels))
if n_at:
    series = population_over_time(labels, n_at)
    series["times"] = [t * DT_NS for t in series["times"]]
    F.population_vs_time(series, ranks, n_motifs, f"{fig_dir}/population.png",
                         title=TITLE)
else:
    print("  skipped the population plot: atoms-per-frame unknown "
          "(set N_ATOMS in config.py)")
print(f"  figures -> {fig_dir}")
