import numpy as np
import scipy.linalg as la
from CorMat.dtw import dtw

class distances():
    """Subpackage of CorMat containing distance functions. Inputs are almost all standardized; this is for
    ease of use with the function CorMat.utils.calculatePairwiseDistances. As such, not every distance requires all
    listed inputs, but it will never hurt to supply them all."""
    # TODO: Refactor distances to use kwargs, and stop listing all of the inputs?
        
    def BuresDistance(A, B, zero_tol=10**-10, *args, **kwargs):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
        For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
        Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else la.fractional_matrix_power(A, 1/2)
        Bhalf = kwargs['Bhalf'] if 'Bhalf' in kwargs.keys() else la.fractional_matrix_power(B, 1/2)
        val = np.trace(A) + np.trace(B) - 2 * distances.faster_RootBuresFidelity(A, B, Ahalf=Ahalf, Bhalf=Bhalf)
        if np.abs(val) > zero_tol:
            return np.sqrt(val).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
        elif np.abs(val) <= zero_tol:
            return 0
    
    def faster_RootBuresFidelity(A, B, *args, **kwargs):
        """Faster implementation of Bures Fidelity"""
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else la.fractional_matrix_power(A, 1/2)
        Bhalf = kwargs['Bhalf'] if 'Bhalf' in kwargs.keys() else la.fractional_matrix_power(B, 1/2)
        return np.sum(np.linalg.svd(Ahalf@Bhalf, compute_uv=False))

    def BuresAngle(A, B, *args, **kwargs):
        """Only applicable to PSD matrices A, B with trace = 1, so that RootBuresFidelity is bounded in [-1,1]"""
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else la.fractional_matrix_power(A, 1/2)
        Bhalf = kwargs['Bhalf'] if 'Bhalf' in kwargs.keys() else la.fractional_matrix_power(B, 1/2)
        val = distances.faster_RootBuresFidelity(A, B, Ahalf=Ahalf, Bhalf=Bhalf)
        return np.arccos(val).real
    
    def Euclidean(A, B, *args, **kwargs):
        return np.linalg.norm(A - B)

    def AffineInvariant(A, B, *args, **kwargs):
        "SPD, not SPSD"
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else la.fractional_matrix_power(A, 1/2)
        A_neghalf = kwargs['A_neghalf'] if 'A_neghalf' in kwargs.keys() else np.linalg.inv(Ahalf)
        return np.linalg.norm(la.logm(A_neghalf @ B @ A_neghalf))
    
    def LogFrobenius(A, B, *args, **kwargs):
        "SPD, not SPSD"
        return np.linalg.norm(la.logm(A) - la.logm(B))
    
    def dtw_distance(A, B, *args, **kwargs):
        """Alias for CorMat.dtw.dtw_distance; placed here for ease of use.
        Not implemented for use with GPU; as such, no matching method is found in CorMat.gpu.distances.
        Instead, this method is optimized with Numba's @jit decorator.
        Default feature distance is Euclidean distance. Enter others with featureDist = distance
        Where distance is a callable and distance(x,y) returns a scalar."""
        DTWfeatureDist = kwargs['DTWfeatureDist'] if 'DTWfeatureDist' in kwargs.keys() else None
        return dtw.dtw_distance(A,B,DTWfeatureDist=DTWfeatureDist)
    
    # def BuresDistance_old(A, B, Ahalf=None, A_neghalf=None, featureDist=None, zero_tol=10**-10):
    #     """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
    #     For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
    #     Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
    #     if Ahalf is None:
    #         Ahalf = la.fractional_matrix_power(A, 1/2)
    #     # val =  np.trace(A) + np.trace(B) - 2 * np.trace(la.fractional_matrix_power(Ahalf@B@Ahalf, 1/2))
    #     val = np.trace(A) + np.trace(B) - 2 * distances.RootBuresFidelity(A,B,Ahalf)
    #     if np.abs(val) > zero_tol:
    #         return np.sqrt(val).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
    #     elif np.abs(val) <= zero_tol:
    #         return 0
        
    # def RootBuresFidelity(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
    #     """Returns the square root of the Bures fidelity between A and B. If tr(a)==tr(B)==1, A and B represent states, then suitable for BuresAngle.
    #     See: https://arxiv.org/pdf/1712.01504.pdf. A Wikipedia article exists as well."""
    #     if Ahalf is None:
    #         Ahalf = la.fractional_matrix_power(A, 1/2)
    #     val = np.trace(la.fractional_matrix_power(Ahalf@B@Ahalf, 1/2))
    #     # val = np.trace(la.fractional_matrix_power(B@A, 1/2)) # Faster than the line above, *tiny* numerical difference
    #     return val
    
    # def BuresAngle_old(A, B, *args, **kwargs):
    #     # Only applicable to PSD matrices with trace = 1
    #     Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else la.fractional_matrix_power(A, 1/2)
    #     A_neghalf = kwargs['A_neghalf'] if 'A_neghalf' in kwargs.keys() else np.linalg.inv(Ahalf)
    #     val = (np.sqrt(distances.BuresFidelity(A, B, Ahalf, A_neghalf)) + 1) % 2 - 1
    #     # val = min(1,max(0, distances.RootBuresFidelity(A, B, Ahalf, A_neghalf).real ))
    #     return np.arccos(val).real
    