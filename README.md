# CorrelationMatrices (CorMat)

A toolbox for analyzing the manifolds which underly timeseries data, especially via analysis of correlation matrices. Furthermore, these tools are intended for use in the clustering and classification of timeseries.

The data pipeline for this package starts with timeseries and ends with the inputs for low-rank manifold embedding methods such as tSNE, MDS, Isomap, etc. Consider using the `sklearn.manifold` or `umap-learn` libraries for such embeddings (many methods exist). This package will (in the future) include utilities for clustering and plotting such embeddings, once they are created.

This package is currently under development, and it may be subject to significant changes in the near future.

## Installation

For now, the only way to install this package is directly from GitHub. One method for doing this is to clone this repository to a local directory, then use `pip` to install that directory as a Python library. From the command line, navigate to the cloned repository, then install via `pip install .`

If you want to modify the library, consider using an editable install via `pip install --editable .`

Once installed, import the library in Python using `import CorMat`

## How to Use CorMat

This package is divided into three primary modules: `utils`, `distances`, and `plots`. There is also `colors`, but this is just a container for a few color sets. Access these using `CorMat.<moduleName>`.

* `utils` contains functions for clipping timeseries, forming correlation matrices, and more. The most important function in `utils` is the function `calculatePairwiseDistances`, which does exactly that. The output of this function is the input for embedding methods such as tSNE, MDS, Isomap, etc.
* `distances` constains distance functions between matrices. The aforementioned `calculatePairwiseDistances` takes a distance function as an input; you can define your own, or use one from this module.
* `plots` is not yet well developed, but will contain functions for creating useful visualizations, such as plotting the low-rank embeddings produced by tSNE, MDS, Isomap, etc.

## GPU Support

CorMat offers support for running distance calculations on a GPU via PyTorch in its `gpu` module. If PyTorch is installed, users can replace any distance `CorMat.distances.<distanceName>` module with `CorMat.gpu.distances.<distanceName>`. This can then be passed as an input to the function `calculatePairwiseDistances` from the `utils` module; just make sure that the matrices passed to the distance function are stored on the GPU (not in RAM), so that the GPU has access to them.

In the future, there will be utility functions added for moving data to/from the GPU. For now, it must be done manually.

The `gpu.linalg` module is not intended for direct use; it is a supplement to `gpu.distances`.

## Example Code and Tests

Currently, no tests or example files are included in this library. The basic usage, however, is as follows:

```
import CorMat

TSes = [list of arrays, each of which is one timeseries]
CMs = [CorMat.utils.TStoCM(TS) for TS in TSes]
pairwiseDistances = calculatePairwiseDistances(CMs, CorMat.distances.<distanceName>)
```

The array `pairwiseDistances` is then used by an embedding method such as tSNE or UMAP. Note that many optional inputs exist for the function `calculatePairwiseDistances`, mostly intended to reduce computation time in certain situations.