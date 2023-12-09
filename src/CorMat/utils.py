import os
import numpy as np

class utils():

    def _embed(data, method, dimension = 2):
        raise NotImplementedError

    def _makeDistFolder(dirName):
        # If no pairwise dists folder exists, make one?
        raise NotImplementedError

    def embedAndPlot(data):
        raise NotImplementedError

    def loadPairwiseDistances(filename):
        print("Attempting to load pairwise distances from file.")
        pairwiseDists = np.loadtxt(filename)
        print("Successfully loaded pairwise distances from file.")
        return pairwiseDists

    def makeEmptyPairwiseDists(numArrays):
        return np.zeros((numArrays, numArrays), dtype=float)

    def calculatePairwiseDistances(Cmats, distFunc, Cmats_half=None, Cmats_neghalf=None):
        # TODO: Make this save progress? Update global variable, or maybe cache parially completed portions?
        # TODO: Implement GPU backend using pytorch (about )
        numArrays = len(Cmats)
        pairwiseDists = utils.makeEmptyPairwiseDists(numArrays)
        
        j=0
        for i in range(numArrays):
            if i%10 == 0:
                print("i =", i, "/", numArrays, "|", "j =", j, "/", numArrays, "          ", end="\r")
            A = Cmats[i]
            if Cmats_half is not None:
                Ahalf = Cmats_half[i]
            else:
                Ahalf = None
            if Cmats_neghalf is not None:
                A_neghalf = Cmats_neghalf[i]
            else:
                A_neghalf = None
            for j in range(0,numArrays):
                if j%30 == 0:
                    print("i =", i, "/", numArrays, "|", "j =", j, "/", numArrays, "          ", end="\r")
                B = Cmats[j]
                dist = distFunc(A, B, Ahalf, A_neghalf)
                pairwiseDists[i,j] = dist
                
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

    def rawTStoClippedCMat(timeseries: np.array, rowsToKeep: int = 90, sampleRate: float = .5, leadingClip: float = 30.0, durationToKeep: float = 300.0, keepAll: bool = False) -> np.array:
        if keepAll: 
            return utils.TStoCM(timeseries)
        else: 
            return utils.TStoCM(utils.clipTS(timeseries[:rowsToKeep, :], sampleRate, leadingClip, durationToKeep))