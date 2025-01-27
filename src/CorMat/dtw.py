import numpy as np
from numba import jit

class dtw():
    """Lots of this implementation inspired by: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html#:~:text=This%20leads%20us%20to%20the,%2Dwarping%20path%7D(5)"""
    
    def compute_cost_matrix(X, Y, DTWfeatureDist=None):
        """Compute the cost matrix of two feature sequences

        Args:
            X (np.ndarray): Sequence 1
            Y (np.ndarray): Sequence 2
            DTWfeatureDist (func(f1, f2)): A distance function that compares two elements of a feature space; 
                                    i.e. compares samples in the time series

        Returns:
            C (np.ndarray): Cost matrix
        """
        if DTWfeatureDist is None:
            def DTWfeatureDist(x,y):
                return np.linalg.norm(x-y)

        height = int(X.shape[0])
        width = int(Y.shape[0])
        C = np.zeros((height, width), dtype=np.float64)
        for i in range(height):
            for j in range(width):
                C[i,j] = DTWfeatureDist(X[i], Y[j])
        return C
    
    @jit(nopython=True)
    def compute_accumulated_cost_matrix(C):
        """Compute the accumulated cost matrix given the cost matrix

        Source: see docstring at file head

        Args:
            C (np.ndarray): Cost matrix

        Returns:
            D (np.ndarray): Accumulated cost matrix
        """
        N = C.shape[0]
        M = C.shape[1]
        D = np.zeros((N, M))
        D[0, 0] = C[0, 0]
        for n in range(1, N):
            D[n, 0] = D[n-1, 0] + C[n, 0]
        for m in range(1, M):
            D[0, m] = D[0, m-1] + C[0, m]
        for n in range(1, N):
            for m in range(1, M):
                D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
        return D
    
    @jit(nopython=True)
    def compute_optimal_warping_path(D):
        """Compute the warping path given an accumulated cost matrix

        Source: see docstring at file head

        Args:
            D (np.ndarray): Accumulated cost matrix

        Returns:
            P (np.ndarray): Optimal warping path
        """
        N = D.shape[0]
        M = D.shape[1]
        n = N - 1
        m = M - 1
        P = [(n, m)]
        while n > 0 or m > 0:
            if n == 0:
                cell = (0, m - 1)
            elif m == 0:
                cell = (n - 1, 0)
            else:
                val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
                if val == D[n-1, m-1]:
                    cell = (n-1, m-1)
                elif val == D[n-1, m]:
                    cell = (n-1, m)
                else:
                    cell = (n, m-1)
            P.append(cell)
            (n, m) = cell
        P.reverse()
        return np.array(P)
    
    def dtw_distance(TS1, TS2, DTWfeatureDist=None):
        """Returns the cost of the optimal warping path given two timeseries TS1, TS2."""
        C = dtw.compute_cost_matrix(TS1, TS2, DTWfeatureDist=DTWfeatureDist)
        D = dtw.compute_accumulated_cost_matrix(C)
        return D[-1,-1]
    
    def simple_optimal_warping_path(TS1, TS2, DTWfeatureDist=None):
        """A wrapper for the three steps of finding the optimal warping path.
        Simply composes the operations of constructing cost matrix, the accumulated cost matrix,
        and backtracking through the accumulated cost matrix."""
        C = dtw.compute_cost_matrix(TS1, TS2, featureDist=DTWfeatureDist)
        D = dtw.compute_accumulated_cost_matrix(C)
        P = dtw.compute_optimal_warping_path(D)
        return P
        