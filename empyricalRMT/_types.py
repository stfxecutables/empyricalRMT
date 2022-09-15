from __future__ import annotations

from enum import Enum
from typing import Callable

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


class MatrixKind(Enum):
    GOE = "goe"
    GUE = "gue"
    GDE = "poisson"
    Poisson = "poisson"
    Uniform = "uniform"

    @classmethod
    def validate(cls, s: str | MatrixKind) -> MatrixKind:
        try:
            if isinstance(s, str):
                return cls(s)
            return s
        except Exception as e:
            values = [e.value for e in cls]
            raise ValueError(f"MatrixKind must be one of {values}") from e
