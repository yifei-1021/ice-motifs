"""
The feature-matrix layout, declared once.

The original code hardcoded this split in at least four places (concat-all.py,
concat-step2.py, concat-soap.ipynb, concat-traj.ipynb), which is why the column
count was ambiguous for so long. Everything here imports from this module.

Canonical trajectory/combined layout, confirmed from the filename produced by
Dataset/split.py --

    TRAJ18058040-HYPO2380017.SOAP-6.0rc.141FRAME.27STEIN.252.SOAP.1VORO.1ENER.20438057.281.npy

and verified empirically against a real split (column 279 mean 32.29 A^3 matching
Table S2, column 280 mean -0.4262 eV/water matching Table S5):

    [  0 : 27 )  Steinhardt   qbar_l, 3 cutoffs x 9 l values
    [ 27 : 279)  SOAP         n_max=8, l_max=6  ->  8*9/2 * 7 = 252
    [279 : 280)  Voronoi cell volume        (regression target 1)
    [280 : 281)  potential energy per water (regression target 2)

Note the *polymorph* matrices are 280 wide (1 VORO, no energy) and a conditional-VAE
variant is 282 wide (+ density). Use LAYOUT_280 / LAYOUT_282 for those.
"""

import numpy as np
from dataclasses import dataclass

STEIN_L = (4, 5, 6, 8, 10, 12, 16, 20, 24)     # 9 values
STEIN_CUTOFFS = (3.2, 4.8, 6.0)                # angstrom, fixed - independent of the SOAP r_cut
N_STEIN = len(STEIN_L) * len(STEIN_CUTOFFS)    # 27

SOAP_N_MAX = 8
SOAP_L_MAX = 6
N_SOAP = SOAP_N_MAX * (SOAP_N_MAX + 1) // 2 * (SOAP_L_MAX + 1)   # 252


@dataclass(frozen=True)
class Layout:
    name: str
    n_stein: int
    n_soap: int
    targets: tuple          # names of the trailing regression columns

    @property
    def stein(self):
        return slice(0, self.n_stein)

    @property
    def soap(self):
        return slice(self.n_stein, self.n_stein + self.n_soap)

    @property
    def targets_slice(self):
        a = self.n_stein + self.n_soap
        return slice(a, a + len(self.targets))

    @property
    def n_features(self):
        return self.n_stein + self.n_soap

    @property
    def n_cols(self):
        return self.n_features + len(self.targets)

    def check(self, array):
        """Raise if `array` does not have this layout's column count."""
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {array.shape}")
        if array.shape[1] != self.n_cols:
            raise ValueError(
                f"{self.name} expects {self.n_cols} columns "
                f"({self.n_stein} Steinhardt + {self.n_soap} SOAP + "
                f"{len(self.targets)} targets: {', '.join(self.targets)}), "
                f"but the array has {array.shape[1]}. "
                f"If this is a 280-column polymorph matrix use LAYOUT_280."
            )
        return array


LAYOUT_281 = Layout("281 (trajectory / combined)", N_STEIN, N_SOAP, ("voronoi_volume", "energy"))
LAYOUT_280 = Layout("280 (polymorph)",             N_STEIN, N_SOAP, ("voronoi_volume",))
LAYOUT_282 = Layout("282 (conditional VAE)",       N_STEIN, N_SOAP, ("voronoi_volume", "energy", "density"))

DEFAULT = LAYOUT_281

BY_NCOLS = {l.n_cols: l for l in (LAYOUT_280, LAYOUT_281, LAYOUT_282)}


def infer(array_or_ncols):
    """Pick the layout from the column count.

    Accepts either a 2-D array or the column count itself -- both spellings turn
    up in calling code and getting it wrong used to raise a confusing
    AttributeError deep inside this function.
    """
    n = (array_or_ncols if isinstance(array_or_ncols, (int, np.integer))
         else array_or_ncols.shape[1])
    try:
        return BY_NCOLS[int(n)]
    except KeyError:
        raise ValueError(
            f"no known layout has {n} columns; known: {sorted(BY_NCOLS)}"
        ) from None
