import platform

import numpy as np

EXPECTED_GOE_VARIANCE = 0.286
EXPECTED_GOE_MEAN = 1.000

DEFAULT_POLY_DEGREE = 7
DEFAULT_SPLINE_SMOOTH = 1.4
DEFAULT_SPLINE_DEGREE = 3

DEFAULT_POLY_DEGREES = [3, 4, 5, 6, 7, 8, 9, 10, 11]
DEFAULT_SPLINE_SMOOTHS = [0.4, 0.5, 0.6] + list(np.linspace(0.7, 0.95, 16))
DEFAULT_SPLINE_DEGREES = [3]

# https://en.wikipedia.org/wiki/ANSI_escape_code#C0_control_codes
if platform.system().lower() == "windows":
    PROG = "Progress:"
    # Don't print too many lines in Windows
    PROG_FREQUENCY = 20
    CONVERG_PROG_INTERVAL = 20
    LEVELVAR_PROG = "Level-variance progress:"
    RIGIDITY_PROG = "Spectral-rigidity progress:"
    CONVERG_PROG = "Finding max iters: "
    PERCENT = "%"
else:
    PROG = "\033[2K Progress:"  # clear line
    PROG_FREQUENCY = 50
    # even at an absurd L=400, we reach convergence at tol=0.001 in 2e6 steps
    # for the rigidity of a large 10000 x 10000 GOE matrix. However for normal
    # L-values, like up to 100 at most, we will not need this many even on a
    # smaller matrix.
    CONVERG_PROG_INTERVAL = int(1e5) // 50
    LEVELVAR_PROG = "\033[2K Level-variance progress:"
    RIGIDITY_PROG = "\033[2K Spectral-rigidity progress:"
    CONVERG_PROG = "\033[2K Finding max iters:"
    PERCENT = "\033[1D%\033[1A"  # go back one to clear space, then up one to undo newline
    ITER_COUNT = "\033[1D\033[1A"  # go back one to clear space, then up one to undo newline

MIN_ITERS = 100
RIGIDITY_GRID = 100
AUTO_MAX_ITERS = int(1e9)
