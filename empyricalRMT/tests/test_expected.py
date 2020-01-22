import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import seaborn as sbn

from pathlib import Path
from statsmodels.nonparametric.kde import KDEUnivariate as KDE

import empyricalRMT.rmt.expected as expected

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.plot import spacings as plotSpacings
from empyricalRMT.rmt.unfold import Unfolder


@pytest.mark.expected
def test_GOE():
    KDE_POINTS = 1000
    for i in range(1):
        M = generateGOEMatrix(1000)
        eigs = np.sort(np.linalg.eigvals(M))
        unfolded = Unfolder(eigs).unfold(trim=False)
        plotSpacings(unfolded, bins=20, kde=True, mode="block")


test_GOE()
