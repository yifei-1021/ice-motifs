"""
Per-system definitions: how to read a trajectory, and what the descriptor
cutoffs should be for that system.

WHY THIS MODULE EXISTS
----------------------
Supplementary Section 9 reapplies the whole framework to systems that are not
ML-BOP water. Two things change between systems and nothing else should:

  1. the FILE FORMAT. The published run read extended-xyz; the new mW and
     silicon runs are LAMMPS dump files.

  2. the LENGTH SCALE. Silicon's first-neighbour distance is ~2.35 A against
     ~2.76 A for coarse-grained water. Applying the water cutoffs unchanged to
     silicon would compare the two systems at the same ABSOLUTE radius, which
     puts a different number of coordination shells inside the cutoff for each
     - so any difference found would be a trivial consequence of the cutoff,
     not a statement about local structure. Cutoffs are therefore expressed as
     MULTIPLES of the first-neighbour distance and converted per system.

The multiples are fixed by the published water run:

    r_cut(SOAP)      6.0 A / 2.76 A = 2.174 d_nn
    Steinhardt       3.2, 4.8, 6.0 A  ->  1.159, 1.739, 2.174 d_nn
    ring network     3.2 A            ->  1.159 d_nn
    coordination     6.0 A            ->  2.174 d_nn

so water reproduces its own published numbers exactly (see `test_systems.py`)
and every other system is measured at the equivalent shell.

d_nn VALUES AND WHERE THEY COME FROM -- CHECK THESE BEFORE YOU RUN
------------------------------------------------------------------
  mW / ML-BOP water   2.76 A   first peak of g(r) at 298 K
  Stillinger-Weber Si 2.35 A   crystalline diamond-Si nearest neighbour

Both are stated in the literature but neither has been measured on YOUR
trajectories. `measure_d_nn()` below computes it from the first g(r) peak of an
actual frame; run it once per system and put the number in the config. If the
measured value differs from the default by more than a few percent, use the
measured one and say so in the SI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# The published water run, in absolute angstrom.
WATER_D_NN = 2.76
PUBLISHED_SOAP_RCUT = 6.0
PUBLISHED_STEIN_CUTOFFS = (3.2, 4.8, 6.0)
PUBLISHED_RING_CUTOFF = 3.2
PUBLISHED_COORD_CUTOFF = 6.0

# The same cutoffs as multiples of d_nn. These are what transfer between systems.
SOAP_RCUT_RED = PUBLISHED_SOAP_RCUT / WATER_D_NN
STEIN_CUTOFFS_RED = tuple(c / WATER_D_NN for c in PUBLISHED_STEIN_CUTOFFS)
RING_CUTOFF_RED = PUBLISHED_RING_CUTOFF / WATER_D_NN
COORD_CUTOFF_RED = PUBLISHED_COORD_CUTOFF / WATER_D_NN


@dataclass(frozen=True)
class System:
    """Everything that differs between systems, in one place."""

    name: str
    d_nn: float                      # first-neighbour distance, angstrom
    fmt: str                         # ase format string
    symbol: str = "O"                # element written into the ASE frames
    mass: float = 18.015             # g/mol, for the Voronoi density conversion
    energy_column: str | None = None # dump column holding per-atom PE, if any
    type_map: dict = field(default_factory=dict)   # LAMMPS type id -> symbol

    # ---- cutoffs, derived so every system is measured at the same shell ----
    @property
    def soap_rcut(self) -> float:
        return round(SOAP_RCUT_RED * self.d_nn, 4)

    @property
    def stein_cutoffs(self) -> tuple:
        return tuple(round(c * self.d_nn, 4) for c in STEIN_CUTOFFS_RED)

    @property
    def ring_cutoff(self) -> float:
        return round(RING_CUTOFF_RED * self.d_nn, 4)

    @property
    def coord_cutoff(self) -> float:
        return round(COORD_CUTOFF_RED * self.d_nn, 4)

    def describe(self) -> str:
        return (
            f"{self.name}: d_nn = {self.d_nn} A\n"
            f"    SOAP r_cut      {self.soap_rcut} A\n"
            f"    Steinhardt      {self.stein_cutoffs} A\n"
            f"    ring network    {self.ring_cutoff} A\n"
            f"    coordination    {self.coord_cutoff} A"
        )


# --------------------------------------------------------------------------- registry

WATER_MLBOP = System(
    name="water-mlbop",
    d_nn=WATER_D_NN,
    fmt="extxyz",
    symbol="O",
    mass=18.015,
)

WATER_MW = System(
    name="water-mw",
    d_nn=2.76,
    fmt="lammps-dump-text",
    symbol="O",
    mass=18.015,
    energy_column="c_pe",
    type_map={1: "O"},
)

SILICON_SW = System(
    name="silicon-sw",
    d_nn=2.35,
    fmt="lammps-dump-text",
    symbol="Si",
    mass=28.0855,
    energy_column="c_pe",
    type_map={1: "Si"},
)

REGISTRY = {s.name: s for s in (WATER_MLBOP, WATER_MW, SILICON_SW)}


def get(name: str) -> System:
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown system {name!r}; known: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]


# --------------------------------------------------------------------------- d_nn

def measure_d_nn(frame, rmax: float = 5.0, nbins: int = 500) -> float:
    """First peak of g(r) for one frame, in angstrom.

    Run this once per trajectory rather than trusting the table above. It uses
    the minimum-image convention, so it is only valid for a box whose shortest
    edge exceeds 2*rmax - which is checked.
    """
    pos = np.asarray(frame.get_positions(), dtype=float)
    cell = np.asarray(frame.get_cell(), dtype=float)
    L = np.linalg.norm(cell, axis=1)
    if np.any(L < 2 * rmax):
        raise ValueError(
            f"box edges {L} too small for rmax={rmax}; lower rmax"
        )

    inv = np.linalg.inv(cell)
    n = len(pos)
    # chunked to keep the pair matrix out of memory for large frames
    counts = np.zeros(nbins)
    edges = np.linspace(0.0, rmax, nbins + 1)
    step = max(1, 4000 // max(1, n // 4000))
    for i0 in range(0, n, 2000):
        d = pos[i0:i0 + 2000, None, :] - pos[None, :, :]
        f = d @ inv
        f -= np.rint(f)
        d = f @ cell
        r = np.linalg.norm(d, axis=-1).ravel()
        r = r[(r > 1e-6) & (r < rmax)]
        counts += np.histogram(r, bins=edges)[0]

    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * np.pi * centres ** 2 * (edges[1] - edges[0])
    g = counts / np.maximum(shell, 1e-12)

    # Smooth, then take the FIRST peak, not the tallest one. In a diamond
    # lattice the second shell has 12 neighbours against 4 in the first, and
    # 12/r2^2 > 4/r1^2, so the global maximum of g(r) is the SECOND shell.
    # Taking argmax here returns 3.84 A for silicon instead of 2.35 A, which
    # would then rescale every cutoff by 63%.
    k = max(3, nbins // 100)
    kern = np.ones(k) / k
    gs = np.convolve(g, kern, mode="same")

    lo = np.searchsorted(centres, 0.5)
    thresh = 0.25 * gs[lo:].max()
    for i in range(lo + 1, len(gs) - 1):
        if gs[i] >= thresh and gs[i] >= gs[i - 1] and gs[i] > gs[i + 1]:
            return float(centres[i])
    return float(centres[lo + int(np.argmax(gs[lo:]))])
