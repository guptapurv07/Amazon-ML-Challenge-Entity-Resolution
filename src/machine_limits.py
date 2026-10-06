"""CPU limits for the 8 GB M2 Mac this project is developed on.

Import this module FIRST in every script, before numpy / polars / sklearn:
their thread pools read these variables once, when they are first imported.
Set ER_THREADS to change the limit (e.g. ER_THREADS=8 on a bigger machine).
"""
import os

N_THREADS = int(os.environ.get("ER_THREADS", "4"))

for _var in (
    "OMP_NUM_THREADS",         # OpenMP: sklearn HistGradientBoosting, LightGBM
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate (numpy on macOS)
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",      # polars ignores the OMP/BLAS variables
    "RAYON_NUM_THREADS",
):
    os.environ[_var] = str(N_THREADS)
