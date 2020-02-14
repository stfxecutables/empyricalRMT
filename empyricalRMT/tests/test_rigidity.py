import numpy as np
from numpy import ndarray
import pandas as pd
import pytest

from pathlib import Path

import empyricalRMT.rmt.plot

from empyricalRMT.rmt.construct import generateGOEMatrix, generate_eigs
from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.observables.rigidity import spectralRigidity


@pytest.mark.plot
def test_plot_rigidity() -> None:
    # good fit for max_L=50 when using generate_eigs(10000)
    # good fit for max_L=55 when using generate_eigs(20000)
    # not likely to be good fit for max_L beyond 20 for generate_eigs(1000)
    eigs = Eigenvalues(generate_eigs(2000))
    unfolded = eigs.unfold(smoother="poly", degree=9)
    unfolded.plot_spectral_rigidity(max_L=40, c_iters=5000)
