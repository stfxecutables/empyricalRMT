from pathlib import Path

from fmri.downsample import downsample_brain
from fmri.flatten import flatten_4D
from rmt.correlater import compute_correlations
from signalproc.clean import get_signals

def test():
    file = Path("/Users/derek/Antigonish/scalar-fmri/Python/sampledata/sub-cntrl03_task-rest_bold.nii.gz")
    img = downsample_brain(file, factor=0.5)
    ARRAY = img.get_fdata()
    DATA = flatten_4D(ARRAY)
    print(f"Before removing air, DATA.shape: {DATA.shape}")

    NO_AIR = get_signals(DATA)
    print(f"After removing air, NO_AIR.shape: {NO_AIR.shape}")

    compute_correlations(NO_AIR)


test()
