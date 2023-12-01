import numpy as np
import scipy.linalg as la

class Distances():

    def BuresDistance(A, B, Ahalf=None, A_neghalf=None):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops."""
        if Ahalf is None:
            Ahalf = la.fractional_matrix_power(A, 1/2)
        return (np.trace(A) + np.trace(B) - 2*np.trace(Ahalf@B@Ahalf)**(1/2))**(1/2)
    
    def AffineInvariante(A, B, Ahalf=None, A_neghalf=None):
        raise NotImplementedError
    
    def LogFrobenius(A, B, Ahalf=None, A_neghalf=None):
        raise NotImplementedError