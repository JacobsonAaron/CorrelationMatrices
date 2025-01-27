# import os
import numpy as np
from scipy import linalg as la
from typing import Callable, Iterable

class utils():

    def _makeDistFolder(dirName):
        # If no pairwise dists folder exists, make one?
        raise NotImplementedError

    # def loadPairwiseDistances(filename):
    #     print("Attempting to load pairwise distances from file.")
    #     pairwiseDists = np.loadtxt(filename)
    #     print("Successfully loaded pairwise distances from file.")
    #     return pairwiseDists

    def calculatePairwiseDistances(Matrices: Iterable, distance: Callable, 
                                    Mats_half: Iterable = None, Mats_neghalf: Iterable = None, 
                                    DTWfeatureDist=None, 
                                    precomputeHalves: bool = False, precomputeNeghalves: bool = False, 
                                    assumeDistIsSymmetric: bool = False, silent: bool = False):
        """Calculates a matrix of pairwise distances between objects in an iterable."""
        # TODO: Allow this to resume progress if interrupted?
        if Mats_half is None and precomputeHalves == True:
            Mats_half = [la.fractional_matrix_power(mat, 1/2) for mat in Matrices]
        if Mats_neghalf is None and precomputeNeghalves == True:
            Mats_neghalf = [np.linalg.inv(mat) for mat in Mats_half]
        
        numArrays = len(Matrices)
        pairwiseDists = np.zeros((numArrays, numArrays), dtype=float)
        j=0
        for i in range(numArrays):
            innerLoopUpperIdx = i+1 if assumeDistIsSymmetric else numArrays # Loop over full rows or just lower triangle
            if i%10 == 0 and not silent:
                print("i =", i, "/", numArrays, "|", "j =", j, "          ", end="\r")
            A = Matrices[i]
            Ahalf = Mats_half[i] if Mats_half is not None else None
            A_neghalf = Mats_neghalf[i] if Mats_neghalf is not None else None
            for j in range(0,innerLoopUpperIdx):
                if j%30 == 0 and not silent:
                    print("i =", i, "/", numArrays, "|", "j =", j, "          ", end="\r")
                B = Matrices[j]
                Bhalf = Mats_half[j] if Mats_half is not None else None
                dist = distance(A, B, Ahalf=Ahalf, Bhalf=Bhalf, A_neghalf=A_neghalf, DTWfeatureDist=DTWfeatureDist)
                pairwiseDists[i,j] = dist
                if assumeDistIsSymmetric:
                    pairwiseDists[j,i] = dist
        return pairwiseDists

    def TStoCM(timeseries: np.array) -> np.array:
        """Accepts a numpy array containing timeseries data and returns a correlation matrix. Note that numpy's corrcoef() method calculates Pearson's coefficient. 
        This method also symmetrizes to eliminate rounding-based asymmetry.

        Args:
            timeseries (np.array): An m by n numpy array containing timeseries data for many variables. Assumes m rows are variables and n columns are samples.

        Returns:
            np.array: An m by m correlation matrix calculated using the input timeseries.
        """
        matrix = np.corrcoef(timeseries)
        return (matrix + np.transpose(matrix)) / 2

    def getNumericalRank(matrix: np.array, tol: float = 10**-13) -> int:
        """Finds the number of eigenvalues of the input matrix which have absolute value greater than tolerance.

        Args:
            matrix (np.array): Input matrix.
            tol (float, optional): Eigenvalues smaller than this will be considered 0. Defaults to 10**-13.

        Returns:
            int: Quantity of eigenvalues larger in absolute value than tol.
        """
        return np.sum(np.abs(np.linalg.eigvals(matrix)) > tol)

    def clipTS(timeseries: np.array, sampleRate: float = .5, leadingClip: float = 30.0, durationToKeep: float = 300.0) -> np.array:
        """Extracts a portion of an input timeseries. This method assumes that rows are variables and columns are samples over time.
        Default values are chosen to suit ADNI fMRI data.
        
        Example:
        Given a timeseries TS which is 40 x 600 (40 vars, 600 samples), this method (with default parameters) will return the 15th column 
        through the 165th column. I.e. returns TS[:, 15:165], which is 40 x 150.

        Args:
            timeseries (np.array): The timeseries from which a window is to be extracted.
            sampleRate (float, optional): The sample rate of the timeseries, likely in samples per second. Defaults to .5.
            leadingClip (float, optional): The duration of time to cut from the start of the timeseries, likely in seconds. Defaults to 30.0.
            durationToKeep (float, optional): The duration of time to keep, likely in seconds. Defaults to 300.0.

        Returns:
            np.array: A shortened timeseries with samples removed from the beginning and/or end.
        """
        start = int(leadingClip * sampleRate)
        samplesToKeep = int(durationToKeep * sampleRate)
        return timeseries[:, start : start + samplesToKeep] # +1 to include last column

    def extractCortexMAT(matrix: np.array, rowsToKeep: int = 90) -> np.array:
        return matrix[:rowsToKeep, :rowsToKeep]

    def rawTStoClippedCMat(timeseries: np.array, rowsToKeep: int = 90, sampleRate: float = .5, leadingClip: float = 30.0, durationToKeep: float = 300.0, keepAll: bool = True) -> np.array:
        if keepAll: 
            return utils.TStoCM(timeseries)
        else: 
            return utils.TStoCM(utils.clipTS(timeseries[:rowsToKeep, :], sampleRate, leadingClip, durationToKeep))
        
    def reinterpolate_TS(TS, desired_len):
        """Given an input time series TS, constructs a piecewise linear interpolation in time, then samples that interpolation at <desired_len> points.
            The result is an upsampled or downsampled time series based on linear. Assumes time is first axis of TS (i.e. reinterpolates on first axis)."""
        if TS.shape[0] == desired_len:
            return TS
        else:
            current_len = TS.shape[0]
            xnew = np.linspace(0, current_len-1, desired_len)
            if len(TS.shape) == 1:
                interp = np.interp(xnew, range(len(TS)), TS)
                return interp
            else:
                toReturn = np.zeros((desired_len, TS.shape[1]))
                for i in range(TS.shape[1]):
                    toReturn[:,i] = np.interp(xnew, range(len(TS[:,i])), TS[:,i])
                return toReturn