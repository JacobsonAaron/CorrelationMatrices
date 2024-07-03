import torch
from CorMat.gpu.linalg import linalg

class distances():

    def BuresDistance(A, B, Ahalf=None, A_neghalf=None, featureDist=None, zero_tol=10**-10):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
        For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
        Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
        # val = torch.trace(A) + torch.trace(B) - 2 * torch.trace(linalg.fractional_mat_power(Ahalf@B@Ahalf, 1/2))
        val = torch.trace(A) + torch.trace(B) - 2 * distances.RootBuresFidelity(A,B,Ahalf,A_neghalf,featureDist)
        if torch.abs(val) > zero_tol:
            return torch.sqrt(val.to(torch.complex128)).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
        elif torch.abs(val) <= zero_tol:
            return 0
    
    def RootBuresFidelity(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
        if Ahalf is None:
            Ahalf = linalg.fractional_mat_power(A, 1/2)
        val = torch.trace(linalg.fractional_mat_power(Ahalf@(B.to(torch.complex128))@Ahalf, 1/2))
        # val = torch.trace(linalg.fractional_matrix_power(B@A, 1/2))
        return val
        
    def BuresAngle(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
        # Only applicable to normalized matrices?
        # Need the %2 to put numbers in range of arccos; not sure if that's the best way to handle it
        # val = (np.sqrt(Distances.BuresFidelity(A, B, Ahalf, A_neghalf)) + 1) % 2 - 1
        val = min(1,max(0, distances.RootBuresFidelity(A, B, Ahalf, A_neghalf).real ))
        return torch.arccos(val).real

    def AffineInvariant(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
        "SPD, not SPSD"
        if A_neghalf is None:
            # TODO: Implement fractional matrix powers on GPU
            # if Ahalf is None:
            #     Ahalf = torch.linalg.fractional_matrix_power(A, 1/2)
            A_neghalf = torch.linalg.inv(Ahalf)
            temp = A_neghalf @ B @ A_neghalf
            # TODO: Compare speed of SVD log on GPU with scipy log on cpu (which is worse, SVD or GPU/CPU back and forth?)
            matLog = linalg.matrixLog(temp)
        return torch.linalg.norm(matLog)
    
    def LogFrobenius(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
        "SPD, not SPSD"
        logA = linalg.matrixLog(A)
        logB = linalg.matrixLog(B)
        return torch.linalg.norm(logA - logB)
    
    def Euclidean(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
        return torch.linalg.norm(A - B)