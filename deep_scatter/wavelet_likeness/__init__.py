"""Wavelet-likeness metrics of trained deep_scatter filters (run_metrics, plots)."""

import os
import sys

# Thousands of tiny least-squares fits: multithreaded BLAS makes each one far slower.
# Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# deep_scatter/, which holds spatial_scattering.py (the reference initialization) and
# training_logs/.
DS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DS_ROOT not in sys.path:
    sys.path.insert(0, DS_ROOT)
