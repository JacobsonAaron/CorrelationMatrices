__all__ = ["__version__", "Distances", "Plots", "Colors", "Utils", "TEST"] # What gets imported when using from _____ import *

import os

import numpy as np

# Use if files have classes inside; this imports as Object (from within files)
# Allows access as CorMat.plots.Plots or as CorMat.Plots
from .distances import Distances
from .plots import Plots
from .colors import Colors
from .utils import Utils

## This imports files as filename.Object
# from CorMat import distances, plots, colors, utils, testModule


# from ._version import __version__


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")