"""Hartigan & Hartigan's dip statistic, and a bootstrap p-value.

Neither `diptest` nor `statsmodels` is installed here, so this is implemented
rather than skipped -- and therefore VALIDATED rather than trusted. Run
`python3 -m icemotifs.dip` (or selftest()) before believing any p-value; an
earlier version of this file returned 0.0 for every input, including two
clearly separated normals, and only the self-test caught it.

DEFINITION USED. The dip is half the smallest sup-norm distance between the
ECDF and any unimodal CDF:

    dip(F_n) = 1/2 * min_m  max( sup|F_n - GCM(F_n) on [x_0, x_m]|,
                                 sup|F_n - LCM(F_n) on [x_m, x_n]| )

where GCM is the greatest convex minorant and LCM the least concave majorant.
The minimisation over the mode position m is done over a grid rather than every
data point, which is an approximation from above (a coarser grid can only
report a larger dip); `mode_grid` controls it and the value used is recorded.

NULL. The p-value is bootstrapped against the UNIFORM, as Hartigan recommends:
among unimodal distributions the uniform is asymptotically least favourable, so
this is conservative. Read the asymmetry honestly -- a small p is evidence of
multimodality; a large p is absence of evidence against unimodality, not proof
of it. That sentence travels with every reported p.
"""
from __future__ import annotations

import numpy as np

MAX_N = 4000          # ECDF is subsampled above this; the dip is scale-free


def _lower_hull(x, y):
    """Indices of the lower convex hull of points sorted by x."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            # cross product of (b-a) x (i-a); <= 0 means b is not below the line
            if (x[b]-x[a])*(y[i]-y[a]) - (y[b]-y[a])*(x[i]-x[a]) <= 0:
                h.pop()
            else:
                break
        h.append(i)
    return np.array(h)


def _upper_hull(x, y):
    """Indices of the upper concave hull (least concave majorant)."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (x[b]-x[a])*(y[i]-y[a]) - (y[b]-y[a])*(x[i]-x[a]) >= 0:
                h.pop()
            else:
                break
        h.append(i)
    return np.array(h)


def _sup_dev(x, y, hull):
    """sup |y - (piecewise-linear interpolation through hull)| over all points."""
    hx, hy = x[hull], y[hull]
    interp = np.interp(x, hx, hy)
    return float(np.abs(y - interp).max())


def dip(x, mode_grid=64):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return 0.0
    if n > MAX_N:                      # subsample the ECDF, keeping the shape
        idx = np.linspace(0, n - 1, MAX_N).astype(int)
        x = np.sort(x)[idx]
    else:
        x = np.sort(x)
    m = x.size
    if x[0] == x[-1]:
        return 0.0
    y = np.arange(m, dtype=np.float64) / (m - 1)
    cand = np.unique(np.linspace(1, m - 2, min(mode_grid, m - 2)).astype(int))
    best = np.inf
    for k in cand:
        dl = _sup_dev(x[:k+1], y[:k+1], _lower_hull(x[:k+1], y[:k+1]))
        dr = _sup_dev(x[k:], y[k:], _upper_hull(x[k:], y[k:]))
        v = max(dl, dr)
        if v < best:
            best = v
    return float(best / 2.0)


_NULL_CACHE = {}


def null_dips(n, n_boot=500, seed=123456, mode_grid=64):
    """Cached bootstrap null. The dip of a uniform sample depends only on n.

    Every sample larger than MAX_N is subsampled to exactly MAX_N before the
    dip is taken, so in practice one null serves every large test. Without the
    cache a dozen tests recompute the identical distribution a dozen times,
    which dominated the runtime.
    """
    key = (int(n), int(n_boot), int(seed), int(mode_grid))
    if key not in _NULL_CACHE:
        rng = np.random.default_rng(seed)
        _NULL_CACHE[key] = np.array(
            [dip(rng.random(n), mode_grid=mode_grid) for _ in range(n_boot)])
    return _NULL_CACHE[key]


def dip_pvalue(x, n_boot=500, seed=123456, mode_grid=64):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = min(x.size, MAX_N)
    d = dip(x, mode_grid=mode_grid)
    null = null_dips(n, n_boot=n_boot, seed=seed, mode_grid=mode_grid)
    return float(d), float((null >= d).mean()), int(x.size)


def selftest(seed=0, n=1500, n_boot=200):
    rng = np.random.default_rng(seed)
    cases = {
        "normal (unimodal)": rng.normal(0, 1, n),
        "two separated normals": np.concatenate(
            [rng.normal(-4, 0.5, n // 2), rng.normal(4, 0.5, n // 2)]),
        "uniform (the null)": rng.random(n),
        "three separated normals": np.concatenate(
            [rng.normal(-6, 0.4, n // 3), rng.normal(0, 0.4, n // 3),
             rng.normal(6, 0.4, n // 3)]),
    }
    out = {}
    for k, s in cases.items():
        d, p, nn = dip_pvalue(s, n_boot=n_boot, seed=seed)
        out[k] = dict(dip=round(d, 5), p=round(p, 4), n=nn)
    out["PASS"] = bool(out["two separated normals"]["p"] < 0.05 and
                       out["three separated normals"]["p"] < 0.05 and
                       out["normal (unimodal)"]["p"] > 0.05 and
                       out["uniform (the null)"]["p"] > 0.01)
    return out


if __name__ == "__main__":
    import json
    r = selftest()
    print(json.dumps(r, indent=1))
    raise SystemExit(0 if r["PASS"] else 1)
