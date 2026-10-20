"""Palette-reduce the redrawn figures in examples/*/reproduce-figures/.

A matplotlib PNG is 24-bit RGB and almost all of those colours are
antialiasing. Mapping each panel onto its own 256-entry palette takes these
figures to about a third of their size at full resolution -- nothing is
scaled, so no line or label gets softer.

Two refusals keep that from becoming a colour knob:

* **A legend colour may not move.** Every hex in the folder's
  `reference/00-legend-relabel.csv` that the panel actually draws is reserved
  as its own palette entry, pixels are mapped to the nearest entry here rather
  than by the imaging library (whose nearest-colour cache nudged drawn motif
  pixels by up to 6 greylevels), and the result is verified exact before
  anything is written. Median-cut alone loses these outright: a motif drawn as
  one thin KDE line holds too few pixels to earn an entry and comes back with
  its colour merged into a neighbour's. The motif colours are the result, so a
  shifted one is not a compression artefact, it is a wrong figure.
* **A colour ramp may not band.** The property panels are drawn with
  `viridis`, and matplotlib renders a colormap through a 256-entry lookup
  table, so every one of those 256 colours is literally present in the file.
  Fitting a palette by pixel count spends almost nothing on the colorbar --
  it is a few hundred pixels -- and the bar comes back visibly stepped. When
  a panel is found to carry that lookup table, `RAMP` of its entries are
  reserved before the rest of the palette is fitted, and the bar stays
  smooth. A panel that carries none of it, such as a motif map, reserves
  nothing and keeps the whole palette for its own colours.
* **An image with transparency is refused.** The published figures in
  `reference/` have transparent backgrounds. Quantising one would flatten it
  onto a guessed background and alter a published artefact, so only the opaque
  panels this package draws itself are touched.

A panel with 256 colours or fewer -- the colour keys, and anything this tool
has already reduced -- is re-saved on an exact palette, which is lossless.
Running it twice is therefore idempotent in pixels but not in bytes: the second
pass rebuilds the same palette in sorted order, so the file changes while not
one pixel does.

`dpi` is carried over on save: these are 200-dpi figures and a reader who
places one in a document needs that.

The notebooks carry the same panels inline, so the same function rewrites the
stored `image/png` payloads; displayed bytes and written file get the same
treatment.

    python tools/shrink-figure-png.py              # shrink files + notebooks
    python tools/shrink-figure-png.py --check      # report, write nothing
"""
import argparse
import base64
import csv
import glob
import functools
import io
import json
import os
import sys

import numpy as np
from PIL import Image

MEDIANCUT = getattr(Image, "Quantize", Image).MEDIANCUT
NODITHER = getattr(Image, "Dither", Image).NONE

COLORS = 256
RAMP = 128       # reserved entries, SHARED between the ramps a panel carries
RAMP_MIN = 32    # lookup-table colours a panel must show to count as carrying it
# Every colormap these panels draw a bar with. The six-panel sheet uses three:
# viridis for the three scalar properties, terrain_r for time and RdYlBu_r for
# energy. Reserving only viridis left those two bars with 1 and 0 of their 256
# entries -- median-cut spends nothing on a few hundred colorbar pixels, so the
# bar comes back stepped while the panel it labels looks fine.
RAMPS = ("viridis", "terrain_r", "RdYlBu_r")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Refused(Exception):
    """The image is not one this tool may touch, or a legend colour moved."""


def legend_hexes(folder):
    """The motif colours of one example folder, as (r, g, b) tuples."""
    path = os.path.join(folder, "reference", "00-legend-relabel.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            h = (row.get("hex") or "").strip().lstrip("#")
            if len(h) == 6:
                out.append(tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    return out


@functools.lru_cache(maxsize=16)
def ramp_lut(name="viridis", n=256):
    """The 256 colours matplotlib actually renders a colormap through."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(name)
    return [tuple(int(round(v * 255)) for v in cm(i / (n - 1))[:3]) for i in range(n)]


def _palette(rgb, uniq, want, colors):
    """Reserved colours first, then median-cut over the rest of the panel."""
    free = colors - len(want)
    if free < 1:
        raise Refused(f"{len(want)} legend colours in a {colors}-entry palette")
    fitted = rgb.quantize(colors=free, method=MEDIANCUT, dither=NODITHER)
    pal = [list(c) for c in want]
    flat = fitted.getpalette()[:3 * free]
    pal += [flat[i:i + 3] for i in range(0, len(flat), 3)]
    return np.asarray(pal[:colors], dtype=np.int32)


def shrink_bytes(data, legend=(), colors=COLORS):
    """Quantise one PNG. Returns the new bytes, or raises Refused."""
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "PA") and im.getchannel("A").getextrema()[0] != 255:
        raise Refused("image has transparency")
    rgb = im.convert("RGB")
    a = np.asarray(rgb)
    # one uint32 per pixel: np.unique over a 1-D array is ~50x the speed of
    # the same call with axis=0, and these panels are several megapixels
    packed = ((a[..., 0].astype(np.uint32) << 16)
              | (a[..., 1].astype(np.uint32) << 8) | a[..., 2])
    upack, inv = np.unique(packed.reshape(-1), return_inverse=True)
    uniq = np.stack([(upack >> 16) & 255, (upack >> 8) & 255, upack & 255],
                    axis=1).astype(np.int32)
    if len(uniq) <= colors:
        pal = uniq.astype(np.int32)              # exact: lossless
        idx = inv.astype(np.uint8)
    else:
        present = set(map(tuple, uniq.tolist()))
        want = [c for c in legend if c in present]
        found = [n for n in RAMPS
                 if sum(c in present for c in ramp_lut(n)) >= RAMP_MIN]
        if found:
            # the budget is SHARED: three bars at 128 entries each would not fit
            # in a 256-entry palette, and the panel still needs entries of its own
            budget = max(1, min(RAMP, colors - len(want) - 16))
            per = max(8, budget // len(found))
            for name in found:
                lut = ramp_lut(name)
                step = max(1, len(lut) // per)
                want += [c for c in lut[::step] if c not in want]
        pal = _palette(rgb, uniq, want, colors)
        # nearest palette entry per DISTINCT colour, then back through `inv`;
        # a pixel equal to a reserved colour has distance 0 to it, and the
        # reserved entries come first, so ties cannot take it elsewhere
        near = np.empty(len(uniq), dtype=np.uint8)
        for i in range(0, len(uniq), 8192):          # bound the (U, 256, 3) temporary
            blk = uniq[i:i + 8192]
            d = ((blk[:, None, :] - pal[None, :, :]) ** 2).sum(2)
            near[i:i + 8192] = d.argmin(1).astype(np.uint8)
        idx = near[inv]
        out = pal[idx].reshape(a.shape)
        for c in want:
            m = (a == np.asarray(c)).all(axis=2)
            if m.any() and not (out[m] == np.asarray(c)).all():
                raise Refused("reserved colour moved: #%02x%02x%02x" % c)
    q = Image.fromarray(idx.reshape(a.shape[:2]), mode="P")
    flat = pal.reshape(-1).tolist()
    q.putpalette(flat + [0] * (768 - len(flat)))
    buf = io.BytesIO()
    save = {"optimize": True}
    if im.info.get("dpi"):
        save["dpi"] = im.info["dpi"]
    q.save(buf, format="PNG", **save)
    return buf.getvalue()


def shrink_file(path, legend=(), write=True):
    with open(path, "rb") as fh:
        data = fh.read()
    new = shrink_bytes(data, legend)
    if write and len(new) < len(data):
        with open(path, "wb") as fh:
            fh.write(new)
    return len(data), min(len(data), len(new))


def shrink_notebook(path, legend=(), write=True):
    """Rewrite the stored image/png outputs of one notebook in place."""
    with open(path) as fh:
        nb = json.load(fh)
    before = after = n = 0
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            b64 = (out.get("data") or {}).get("image/png")
            if not b64:
                continue
            raw = base64.b64decode(b64)
            new = shrink_bytes(raw, legend)
            before += len(raw)
            after += min(len(raw), len(new))
            n += 1
            if write and len(new) < len(raw):
                out["data"]["image/png"] = base64.b64encode(new).decode("ascii")
    if write and n:
        with open(path, "w") as fh:
            json.dump(nb, fh, indent=1, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
    return before, after, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--only", metavar="FOLDER", default=None,
                    help="one example folder name")
    a = ap.parse_args(argv)
    write = not a.check
    folders = sorted(d for d in glob.glob(os.path.join(HERE, "examples", "*"))
                     if os.path.isdir(d) and (not a.only
                                              or os.path.basename(d) == a.only))
    tot_b = tot_a = 0
    for folder in folders:
        pngs = sorted(glob.glob(os.path.join(folder, "reproduce-figures", "*.png")))
        nb = os.path.join(folder, "analysis.ipynb")
        if not pngs and not os.path.exists(nb):
            continue
        legend = legend_hexes(folder)
        fb = fa = 0
        for p in pngs:
            b, n = shrink_file(p, legend, write)
            fb += b
            fa += n
        nb_b = nb_a = nb_n = 0
        if os.path.exists(nb):
            nb_b, nb_a, nb_n = shrink_notebook(nb, legend, write)
        print(f"  {os.path.basename(folder):16s} {len(pngs):2d} png "
              f"{fb / 1048576:5.2f} -> {fa / 1048576:5.2f} MiB   "
              f"{nb_n:2d} inline {nb_b / 1048576:5.2f} -> {nb_a / 1048576:5.2f} MiB")
        tot_b += fb + nb_b
        tot_a += fa + nb_a
    print(f"  {'total':16s} {tot_b / 1048576:21.2f} -> {tot_a / 1048576:5.2f} MiB"
          + ("   (--check: nothing written)" if a.check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
