import platform

import numpy as np

EXPECTED_GOE_VARIANCE = 0.286
EXPECTED_GOE_MEAN = 1.000

DEFAULT_POLY_DEGREE = 9
DEFAULT_SPLINE_SMOOTH = 1.4
DEFAULT_SPLINE_DEGREE = 3

DEFAULT_POLY_DEGREES = [3, 4, 5, 6, 7, 8, 9, 10, 11]
DEFAULT_SPLINE_SMOOTHS = [0.4, 0.5, 0.6] + list(np.linspace(0.7, 0.95, 16))
DEFAULT_SPLINE_DEGREES = [3]

# https://en.wikipedia.org/wiki/ANSI_escape_code#C0_control_codes
if platform.system().lower() == "windows":
    PROG = "Progress:"
    PROG_FREQUENCY = 20  # Don't print too many lines in Windows
    LEVELVAR_PROG = "Level-variance progress:"
    RIGIDITY_PROG = "Spectral-rigidity progress:"
    PERCENT = "%"
else:
    PROG = "\033[2KProgress:"  # clear line
    PROG_FREQUENCY = 50
    LEVELVAR_PROG = "\033[2K Level-variance progress:"
    RIGIDITY_PROG = "\033[2K Spectral-rigidity progress:"
    PERCENT = "\033[1D%\033[1A"  # go back one to clear space, then up one to undo newline