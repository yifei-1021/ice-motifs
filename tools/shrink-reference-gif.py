"""Downsample an OVITO motif GIF for the examples/ reference folders.

Two things here are not optional:

* ONE GLOBAL palette for every frame. A per-frame adaptive palette shifts a
  motif's hue between frames, which destroys the colour key the figures rely on.
* The palette is built from frames spanning the WHOLE trajectory. Building it
  from the opening frames gives a palette with no entry for motifs that only
  appear once the nucleus has grown, and they get mapped onto whatever is
  nearest -- blue nuclei came out olive.

Dithering is off on purpose: these are flat-shaded spheres, and Floyd-Steinberg
scatters black speckle over them instead of smoothing anything.
Output frame duration is scaled by the frame step, so wall-clock pacing -- and
so the apparent transition rate -- is preserved.
"""
import io
import os
import sys

import numpy as np
from PIL import Image, ImageSequence

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
MEDIANCUT = getattr(Image, "Quantize", Image).MEDIANCUT
NODITHER = getattr(Image, "Dither", Image).NONE


def _flatten(src, px):
    im = Image.open(src)
    dur = im.info.get("duration", 100)
    out = []
    for fr in ImageSequence.Iterator(im):
        bg = Image.new("RGB", fr.size, (255, 255, 255))
        fr = fr.convert("RGBA")
        bg.paste(fr, (0, 0), fr)
        out.append(bg.resize((px, px), RESAMPLE))
    return out, dur


def shrink(src, dst, px=260, step=3, colors=128, n_ref=24):
    flat, dur = _flatten(src, px)
    kept = flat[::step]
    idx = [round(i * (len(kept) - 1) / (n_ref - 1)) for i in range(min(n_ref, len(kept)))]
    idx = sorted(set(idx))
    ref = Image.new("RGB", (px, px * len(idx)))
    for i, j in enumerate(idx):
        ref.paste(kept[j], (0, i * px))
    pal = ref.quantize(colors=colors, method=MEDIANCUT)
    q = [f.quantize(palette=pal, dither=NODITHER) for f in kept]
    q[0].save(dst, save_all=True, append_images=q[1:], loop=0,
              duration=dur * step, optimize=True, disposal=2)
    return len(q), dur * step


def _colour_set(frames):
    """Every RGB value the whole animation uses, as a (<=256, 3) array."""
    a = np.concatenate([np.asarray(f).reshape(-1, 3) for f in frames])
    packed = ((a[:, 0].astype(np.uint32) << 16)
              | (a[:, 1].astype(np.uint32) << 8) | a[:, 2])
    u = np.unique(packed)
    if len(u) > 256:
        raise ValueError("source uses %d colours; this mode expects a GIF" % len(u))
    return np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.int32)


def rescale(src, dst, px):
    """Re-scale an already-reduced GIF, keeping its own colour set exactly.

    For the second pass -- the shipped movie down to the width the README shows
    it at. The palette is NOT rebuilt: the colours present in the source are
    reused as the target palette, so every output pixel is a colour the source
    already had and no motif hue can drift. Resampling is the only loss.
    Frame count and frame duration are untouched, so the pacing is preserved.
    """
    im = Image.open(src)
    dur = im.info.get("duration", 100)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    pal = _colour_set(frames)
    ref = Image.new("P", (1, 1))
    flat = pal.reshape(-1).tolist()
    ref.putpalette(flat + [0] * (768 - len(flat)))
    out = [f.resize((px, px), RESAMPLE).quantize(palette=ref, dither=NODITHER)
           for f in frames]
    buf = io.BytesIO()
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:], loop=0,
                duration=dur, optimize=True, disposal=2)
    data = buf.getvalue()
    # the colours of the result must be a subset of the source's
    chk = Image.open(io.BytesIO(data))
    got = _colour_set([f.convert("RGB") for f in ImageSequence.Iterator(chk)])
    have = set(map(tuple, pal.tolist()))
    stray = [tuple(c) for c in got.tolist() if tuple(c) not in have]
    if stray:
        raise ValueError("introduced %d colour(s) the source did not have" % len(stray))
    if chk.n_frames != len(frames) or chk.info.get("duration") != dur:
        raise ValueError("frame count or duration changed")
    with open(dst, "wb") as fh:
        fh.write(data)
    return len(frames), dur


if __name__ == "__main__":
    if sys.argv[1] == "--rescale":          # second pass, colours kept
        src, dst, px = sys.argv[2], sys.argv[3], int(sys.argv[4])
        n, d = rescale(src, dst, px)
    else:                                   # first pass, from the OVITO render
        src, dst = sys.argv[1], sys.argv[2]
        px, step, colors = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        n, d = shrink(src, dst, px, step, colors)
    print(f"  {os.path.basename(dst):30s} {os.path.getsize(dst)/1048576:5.2f} MB  "
          f"{n} frames @ {d} ms")

# Used to build examples/<dataset>/reference/motif-evolution.gif from the
# author's OVITO renders:
#
#   python tools/shrink-reference-gif.py in.gif out.gif 260 3 128
#
# and then down to the width the README displays them at, keeping the colour
# set that first pass chose:
#
#   python tools/shrink-reference-gif.py --rescale in.gif out.gif 185
#
# Check the result by eye against the source at two timepoints. Size is the
# same whether the palette is right or wrong, so size is not the check.
