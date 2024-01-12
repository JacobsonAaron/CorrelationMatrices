__all__ = ["__version__", "linalg", "distances"]

import os

# Use if files have classes inside; this imports as Object (from within files)
# Allows access as CorMat.plots.Plots or as CorMat.Plots
from .linalg import linalg
from .distances import distances