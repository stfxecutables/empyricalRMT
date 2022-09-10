from enum import Enum
from typing import Callable, Literal, Union

from numba.extending import as_numba_type
from numpy import bool_, floating, signedinteger, unsignedinteger
from numpy.typing import NDArray

PyInt = as_numba_type(int)
PyFloat = as_numba_type(float)
PyBool = as_numba_type(bool)


fArr = NDArray[floating]
iArr = NDArray[signedinteger]
uArr = NDArray[unsignedinteger]
bArr = NDArray[bool_]

Smoother = Callable[[fArr], fArr]

MatrixKind = Literal["goe", "gue", "gde", "uniform", "poisson"]
