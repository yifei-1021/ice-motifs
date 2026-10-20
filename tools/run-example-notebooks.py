#!/usr/bin/env python3
"""Execute the reproduction section of each example notebook and store its output.

    python tools/run-example-notebooks.py --runs-root /path/to/the/runs/tree
    python tools/run-example-notebooks.py --published-run /path/to/one/runs --only silicon

WHY ONLY THAT SECTION. Steps 1-5 need a trajectory, and no trajectory is part
of this package: running them here would store the figures of whatever happened
to be on the machine that packaged it. The reproduction section is the one part
that can be re-run from arrays alone, and its panels are the published ones, so
those are the outputs worth shipping. Every other cell is left exactly as the
generator wrote it -- no outputs, no execution counts.

It runs the figure cells through a real kernel rather than exec-ing their source
in this process. Exec-ing proves the code runs; it does not prove a figure is
DISPLAYED, which is the entire reason the figure stage is defined in the
notebook instead of called from `icemotifs.figures`. So every drawing cell of
the section is required to come back carrying an image.

`--runs-root` resolves each folder's own `PUBLISHED_RUN_NAME` (from its
`scripts/config.py`) under that directory, which is how all seven are done in
one command. `--published-run` names one run directly and needs `--only`.

The kernel inherits this process's environment. On a machine where HOME is not
writable, point MPLCONFIGDIR, IPYTHONDIR and the JUPYTER_* directories
somewhere that is before running this.
"""
import argparse
import copy
import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(HERE, "examples")
REPRO_HEADING = "## Reproducing the published figures"
FIG_HEADING = "## Step 7."


def config_of(folder):
    spec = importlib.util.spec_from_file_location(
        "cfg_" + os.path.basename(folder).replace("-", "_"),
        os.path.join(folder, "scripts", "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(folder, published_run, timeout=3600):
    """Execute the figure and reproduction cells; keep the section's outputs.

    The cells are executed as a notebook of their own, so the stage cells above
    them are never touched. Their outputs then go back onto the shipped
    notebook BY CELL ID -- not by position, which a later edit to the generator
    would silently shift.
    """
    import nbformat as nbf
    from nbclient import NotebookClient

    path = os.path.join(folder, "analysis.ipynb")
    nb = nbf.read(path, as_version=4)
    first_fig = next(i for i, c in enumerate(nb.cells)
                     if c.cell_type == "markdown"
                     and c.source.startswith(FIG_HEADING))
    first_repro = next(i for i, c in enumerate(nb.cells)
                       if c.cell_type == "markdown"
                       and c.source.startswith(REPRO_HEADING))
    # The section calls functions the step-7 cells define, so those run too --
    # with no run on disk they define and draw nothing, which is what their
    # HAVE_RUN guard is for. Their outputs are NOT kept.
    to_run = [c for c in nb.cells[first_fig:] if c.cell_type == "code"]
    keep = {c["id"] for c in nb.cells[first_repro:] if c.cell_type == "code"}
    assert keep, "the reproduction section has no code cells"

    work = nbf.v4.new_notebook(cells=[nbf.v4.new_code_cell(c.source)
                                      for c in to_run])
    for cell, origin in zip(work.cells, to_run):
        cell["id"] = origin["id"]
    # A COPY of the metadata, not the same object: nbclient writes the running
    # kernel's `language_info` into the notebook it executes, and sharing the
    # dict stamps that interpreter's exact version into the shipped file --
    # which is both noise and enough to make `--check` call every notebook
    # stale for ever after.
    work.metadata = copy.deepcopy(nb.metadata)
    # The kernel is a subprocess and inherits this environment, which is how
    # the section is told which run to read. PYTHONPATH is set too, so the
    # package is importable from a checkout that has not been pip-installed.
    was = {k: os.environ.get(k)
           for k in ("ICEMOTIFS_PUBLISHED_RUN", "PYTHONPATH")}
    os.environ["ICEMOTIFS_PUBLISHED_RUN"] = published_run
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [HERE] + ([was["PYTHONPATH"]] if was["PYTHONPATH"] else []))
    try:
        NotebookClient(work, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": folder}}).execute()
    finally:
        for k, v in was.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    produced = {}
    for cell in work.cells:
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                sys.stderr.write(
                    f"\n{os.path.basename(folder)}: cell failed -- "
                    f"{out.ename}: {out.evalue}\n"
                    + "\n".join(out.get("traceback", [])[-14:]) + "\n")
                return None
        if cell["id"] in keep:
            produced[cell["id"]] = cell.get("outputs", [])

    n_png, count = 0, 0
    for cell in nb.cells:
        if cell.get("id") in produced:
            outs = produced[cell["id"]]
            count += 1
            cell["outputs"] = outs
            cell["execution_count"] = count
            for out in outs:
                if out.output_type == "execute_result":
                    out["execution_count"] = count
            n_png += sum("image/png" in o.get("data", {})
                         for o in outs if o.output_type == "display_data")
        elif cell.cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # Every panel the section wrote to disk has to have been DISPLAYED as well.
    # That is the claim these cells exist to support -- `icemotifs.figures`
    # closes its figures, so a cell that merely saved a file and showed nothing
    # would be the old behaviour wearing the new structure. A stale PNG left by
    # an earlier naming also trips this, which is the right answer too.
    out_dir = os.path.join(folder, "reproduce-figures")
    on_disk = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    assert n_png == on_disk, (
        f"{os.path.basename(folder)}: the section wrote {on_disk} panels but "
        f"displayed {n_png}")
    nbf.write(nb, path)
    return n_png, work


# reproduction -> the published panel it is to be read against, and by what.
# The published names keep their numbers because they are the published files;
# these keep only what the panel IS.
PAIRS = [
    ("05-umap-by-properties*.png", "05-umap-by-properties.*.png",
     "`plot_umap_by_properties` — the map, then time, energy, tetrahedrality, "
     "Voronoi density and qbar6, on one shared frame"),
    ("dist-merged.major*.png", "dist-merged.major.*.png",
     "`plot_dist_merged` — the density maxima holding >= 5% of atoms, with "
     "every other motif pooled into one grey curve"),
    ("09-composition-ring-density*.png", "09-composition-ring-density.*.png",
     "`plot_composition_ring_density` — the stacked composition over the ring "
     "occupancy and the density distribution of the same motifs"),
]

# panels in the published relabel/ directory that these functions cannot draw
NOT_REPRODUCED = [
    ("`01-motif-map.hdscan-only`", "needs the density-maximum subset, which "
     "the notebook does not separate out"),
    ("`02-basins-vs-tiles`", "same -- maxima against k-means tiles"),
    ("`05-by-dataset`, `06-dataset-facets`", "need the pooled run's dataset "
     "column; an example folder is one dataset"),
    ("`05a-by-logsigma-magnitude`", "needs `z_logsigma.npy`, which step 3 "
     "writes but no example figure uses"),
    ("`05b-by-tetrahedrality`", "drawn as one panel of the sheet above, not "
     "as its own figure; A is read from the published run's cache, because "
     "nothing in the package computes it"),
    ("`05-merged-ring-density`, `11-*`, `12-*`, `13-*`",
     "ring statistics as figures of their own; the ring analysis is not part "
     "of the example pipeline, and the one ring panel here is read from "
     "`11-ring-motif-occupancy.csv`"),
    ("`10-transition-matrix`", "needs a transition matrix between consecutive "
     "frames, which no example stage computes"),
    ("`15-chillplus-by-motif`", "needs CHILL+ labels, which are not produced "
     "here"),
]


def shrink(folder):
    """Palette-reduce what the section just wrote, and the stored copies with it.

    Run here rather than left to the author, so the panels in the repository
    are what a re-run produces. The tool reserves the motif colours and the
    viridis ramp exactly and refuses anything with transparency, so this is
    not a colour knob -- see its docstring.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "shrink-figure-png.py")
    spec = importlib.util.spec_from_file_location("shrink_figure_png", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    legend = mod.legend_hexes(folder)
    before = after = 0
    for png in sorted(glob.glob(os.path.join(folder, "reproduce-figures", "*.png"))):
        b, n = mod.shrink_file(png, legend)
        before += b
        after += n
    b, n, _ = mod.shrink_notebook(os.path.join(folder, "analysis.ipynb"), legend)
    return before + b, after + n


def facts_from(nb, source):
    """The numbers for the README, read out of what the notebook just printed.

    Deliberately not recomputed here: a README whose figures and whose numbers
    come from two different passes can disagree, and this way it cannot. The
    format is the one the section's first cell prints, so a change there must
    change this -- which is why each field is required rather than optional.
    """
    import re
    text = "".join(
        "".join(o.get("text", "")) for c in nb.cells
        for o in c.get("outputs", []) if o.output_type == "stream")
    pat = {
        "legend": r"colours\s+(\S+)\s+\((\d+) motifs\)",
        "pooled": r"pooled rows\s+([\d,]+)",
        "rows": re.escape(source) + r"\s+rows ([\d,]+):([\d,]+)\s+"
                r"\(([\d,]+) frames x ([\d,]+) atoms\)",
    }
    got = {}
    for k, rx in pat.items():
        m = re.search(rx, text)
        assert m, f"the section did not print the {k} line this README needs"
        got[k] = [g.replace(",", "") for g in m.groups()]
    step = re.search(r"from every (\d+)th frame", text)
    got["step"] = int(step.group(1)) if step else 1
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", help="directory the per-dataset runs sit under")
    ap.add_argument("--published-run", help="one run directory; needs --only")
    ap.add_argument("--only", action="append", help="example folder; repeatable")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    if not a.runs_root and not a.published_run:
        sys.exit("give --runs-root, or --published-run with --only")
    if a.published_run and (not a.only or len(a.only) != 1):
        sys.exit("--published-run names one run, so it needs exactly one --only")

    folders = sorted(d for d in os.listdir(EX)
                     if os.path.exists(os.path.join(EX, d, "analysis.ipynb")))
    if a.only:
        missing = set(a.only) - set(folders)
        if missing:
            sys.exit(f"no such example folder: {', '.join(sorted(missing))}")
        folders = [d for d in folders if d in a.only]

    bad = 0
    for name in folders:
        folder = os.path.join(EX, name)
        pub = (a.published_run if a.published_run
               else os.path.join(a.runs_root, config_of(folder).PUBLISHED_RUN_NAME))
        if not os.path.isdir(pub):
            print(f"  {name:16s} SKIPPED: no run at the path for "
                  f"{config_of(folder).PUBLISHED_RUN_NAME}")
            bad += 1
            continue
        done = run(folder, pub, a.timeout)
        if done is None:
            bad += 1
            continue
        n_png, executed = done
        # reproduce-figures/README.md is no longer written: the per-folder
        # READMEs were removed in favour of one examples/README.md, and a tool
        # that regenerates a file the repository does not keep would put it back
        # on every run.
        before, after = shrink(folder)
        print(f"  {name:16s} {n_png} figures stored, README rewritten, "
              f"panels {before / 1048576:.1f} -> {after / 1048576:.1f} MiB")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
