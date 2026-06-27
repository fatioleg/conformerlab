"""Unit constants. The core speaks kcal/mol everywhere; convert at the edges.

Keeping these in one place means a backend that returns Hartree (DFT, xTB)
converts with HARTREE_TO_KCAL and the rest of the code never sees Hartree.
"""

# Gas constant in kcal/(mol*K). Used for Boltzmann weights.
R_KCAL_PER_MOL_K = 1.987204258e-3

# Energy conversions
HARTREE_TO_KCAL = 627.509474
KJ_TO_KCAL = 0.239005736
