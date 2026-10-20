"""6. Tabulate the motifs: population and mean properties.

    python scripts/06_analyze.py

Writes `motif_table.csv`: one row per motif holding at least 1% of the atoms --
its population, its `motif_relabel` rank, and the mean of each per-atom property
the run has. Lower `min_percent` in the call below to see the rest. This is the
table step 7 draws from, and the one to read before quoting a motif.

Population is a share of ATOMS. A vocabulary of n motifs where one holds 40% of
them is not n equal things, and the count alone would imply otherwise.
"""
import csv, json, os, sys
import numpy as np
from icemotifs.analysis import motif_table
from config import OUT, need_stage, load_properties

need_stage(f"{OUT}/motifs", "scripts/04_cluster.py")
labels = np.load(f"{OUT}/motifs/labels.npy")
ranks = {int(r["motif"]): int(r["motif_relabel"])
         for r in csv.DictReader(open(f"{OUT}/motifs/relabel.csv"))} \
    if os.path.exists(f"{OUT}/motifs/relabel.csv") else None

props = load_properties(OUT, len(labels))
rows = motif_table(labels, props, ranks=ranks, min_percent=1.0)
if not rows:
    print("no motif holds >=1% of atoms; lower min_percent to inspect", file=sys.stderr)
    sys.exit(1)
out = f"{OUT}/motif_table.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# the COUNT of rows is not printed: with min_percent low it is the motif
# total, which is the density maxima plus a chosen k. Quote clustering.json.
print(f"  motifs holding >=1% of atoms -> {out}")
for r in rows[:6]:
    print(f"    motif {r['motif']:>3}  {r['percent_of_atoms']:>5.1f}%  " +
          "  ".join(f"{k}={r[k]}" for k in props if r.get(k) is not None))
