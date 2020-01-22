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
        spacings = np.sort(unfolded[1:] - unfolded[:-1])
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0)
        plotting_spacings = np.linspace(spacings[0], spacings[-1], len(spacings) - 1)
        evaluated_kde = np.empty_like(plotting_spacings)
        for i, s in enumerate(evaluated_kde):
            evaluated_kde[i] = kde.evaluate(plotting_spacings[i])

        axes = sbn.distplot(
            spacings,
            norm_hist=True,
            bins=20,  # doane
            kde=True,
            axlabel="spacing (s)",
            color="black",
            label="Empirical Spacing Distribution",
            kde_kws={
                "kernel": "gau",
                "label": "Automatic KDE",
                "cut": 0,
                "bw": "scott",
            },
        )

        kde_curve = axes.plot(plotting_spacings, evaluated_kde, label="Manual KDE")
        plt.setp(kde_curve, color="#EA00FF")
        plt.title("Manual vs. Auto KDE")
        plt.legend()
        plt.show()

        # plotSpacings(
        #     actual,
        #     bins=25,
        #     kde=True,
        #     kde_kws={"kernel": "gau", "label": "KDE", "cut": 0, "bw": "scott"},
        # )


test_GOE()
