from numba.extending import as_numba_type
from numpy import bool_
from numpy import float64 as f64
from numpy import int64 as i64
from numpy.typing import NDArray

PyInt = as_numba_type(int)
PyFloat = as_numba_type(float)
PyBool = as_numba_type(bool)
