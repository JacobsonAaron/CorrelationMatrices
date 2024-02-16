import numpy as np
import scipy.linalg as la

class distances():

    def BuresDistance(A, B, Ahalf=None, A_neghalf=None, zero_tol=10**-10):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
        For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
        Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
        if Ahalf is None:
            Ahalf = la.fractional_matrix_power(A, 1/2)
        # val =  np.trace(A) + np.trace(B) - 2 * np.trace(la.fractional_matrix_power(Ahalf@B@Ahalf, 1/2))
        val = np.trace(A) + np.trace(B) - 2 * distances.RootBuresFidelity(A,B)
        if np.abs(val) > zero_tol:
            return np.sqrt(val).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
        elif np.abs(val) <= zero_tol:
            return 0
    
    # def BuresDistanceFromWikipedia(A, B, Ahalf=None, A_neghalf=None, zero_tol=10**-10):
    #     """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
    #     This version of this function mirrors the syntax currently found on Wikipedia."""
    #     if Ahalf is None:
    #         Ahalf = la.fractional_matrix_power(A, 1/2)
    #     val = np.trace(A) + np.trace(B) - 2*np.sqrt(distances.BuresFidelity(A, B, Ahalf))
    #     if np.abs(val) > zero_tol:
    #         return np.sqrt(val).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
    #     elif np.abs(val) <= zero_tol:
    #         return 0
    
    def RootBuresFidelity(A, B, Ahalf=None, A_neghalf=None):
        """Returns the square root of the Bures fidelity between A and B. If tr(a)==tr(B)==1, A and B represent states, suitable for BuresAngle.
        See: https://arxiv.org/pdf/1712.01504.pdf. A Wikipedia article exists as well.
        So far, this always equals 0. Not sure if this is implemented correctly."""
        if Ahalf is None:
            Ahalf = la.fractional_matrix_power(A, 1/2)
        val = np.trace(la.fractional_matrix_power(Ahalf@B@Ahalf, 1/2))
        # val = np.trace(la.fractional_matrix_power(B@A, 1/2)) # Faster than the line above, *tiny* numerical difference
        return val
        
    def BuresAngle(A, B, Ahalf=None, A_neghalf=None):
        # Only applicable to normalized matrices?
        # val = (np.sqrt(Distances.BuresFidelity(A, B, Ahalf, A_neghalf)) + 1) % 2 - 1
        val = min(1,max(0, distances.RootBuresFidelity(A, B, Ahalf, A_neghalf).real ))
        return np.arccos(val).real
    
    def Euclidean(A, B, Ahalf=None, A_neghalf=None):
        return np.linalg.norm(A - B)

    def AffineInvariant(A, B, Ahalf=None, A_neghalf=None):
        "SPD, not SPSD"
        if A_neghalf is None:
            if Ahalf is None:
                Ahalf = la.fractional_matrix_power(A, 1/2)
            A_neghalf = np.linalg.inv(Ahalf)
        return np.linalg.norm(la.logm(A_neghalf @ B @ A_neghalf))
    
    def LogFrobenius(A, B, Ahalf=None, A_neghalf=None):
        "SPD, not SPSD"
        return np.linalg.norm(la.logm(A) - la.logm(B))