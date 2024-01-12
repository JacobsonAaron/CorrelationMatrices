import numpy as np
import scipy.linalg as la

class distances():

    def BuresDistance(A, B, Ahalf=None, A_neghalf=None, zero_tol=10**-10):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
        For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
        Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
        # TODO: Add GPU support. About 5.3 times faster on my computer.
        if Ahalf is None:
            Ahalf = la.fractional_matrix_power(A, 1/2)
        val =  np.trace(A) + np.trace(B) - 2 * np.trace(la.fractional_matrix_power(Ahalf@B@Ahalf, 1/2))
        if np.abs(val) > zero_tol:
            return np.sqrt(val).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
        elif np.abs(val) <= zero_tol:
            return 0
    
    def BuresDistanceFromWikipedia(A, B, Ahalf=None, A_neghalf=None, zero_tol=10**-10):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
        This version of this function mirrors the syntax currently found on Wikipedia."""
        if Ahalf is None:
            Ahalf = la.fractional_matrix_power(A, 1/2)
        val = np.trace(A) + np.trace(B) - 2*np.sqrt(Distances.BuresFidelity(A, B, Ahalf))
        if np.abs(val) > zero_tol:
            return np.sqrt(val).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
        elif np.abs(val) <= zero_tol:
            return 0
    
    def BuresFidelity(A, B, Ahalf=None, A_neghalf=None):
        if Ahalf is None:
            Ahalf = la.fractional_matrix_power(A, 1/2)
        val = np.trace(la.fractional_matrix_power(Ahalf@B@Ahalf, 1/2)).real
        return val**2
        
    def BuresAngle(A, B, Ahalf=None, A_neghalf=None):
        # Only applicable to normalized matrices?
        # Need the %2 to put numbers in range of arccos; not sure if that's the best way to handle it
        # val = (np.sqrt(Distances.BuresFidelity(A, B, Ahalf, A_neghalf)) + 1) % 2 - 1
        val = np.sqrt(Distances.BuresFidelity(A, B, Ahalf, A_neghalf))
        return np.arccos(val)
    
    def Euclidean(A, B, Ahalf=None, A_neghalf=None):
        return np.norm(A - B)

    def AffineInvariante(A, B, Ahalf=None, A_neghalf=None):
        raise NotImplementedError
    
    def LogFrobenius(A, B, Ahalf=None, A_neghalf=None):
        raise NotImplementedError