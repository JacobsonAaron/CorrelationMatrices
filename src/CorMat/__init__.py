__all__ = ["__version__", "distances", "plots", "colors", "utils", "dtw", "gpu"] # What gets imported when using from _____ import *
# TODO: Fix the __version__ attribute. Not sure what is wrong with it

# Use if files have classes inside; this imports as Object (from within files)
# Allows access as CorMat.plots.Plots or as CorMat.Plots
from .distances import distances
from .plots import plots
from .colors import colors
from .utils import utils
from .dtw import dtw
from CorMat import gpu

## This imports files as filename.Object
# from CorMat import distances, plots, colors, utils, testModule


# from ._version import __version__


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")