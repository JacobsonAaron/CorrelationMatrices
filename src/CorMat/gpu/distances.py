import torch
from CorMat.gpu.linalg import linalg

class distances():

    def BuresDistance(A, B, zero_tol=10**-10, *args, **kwargs):
        """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
        For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
        Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else linalg.fractional_matrix_power(A, 1/2)
        Bhalf = kwargs['Bhalf'] if 'Bhalf' in kwargs.keys() else linalg.fractional_matrix_power(B, 1/2)
        fastMode = kwargs['fastMode'] if 'fastMode' in kwargs.keys() else False
        val = torch.trace(A) + torch.trace(B) - 2 * distances.faster_RootBuresFidelity(A, B, Ahalf=Ahalf, Bhalf=Bhalf, fastMode=fastMode)
        if val.real >= 0:
            return torch.sqrt(val.real) # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
        elif torch.abs(val) < zero_tol:
            return 0
        elif torch.abs(val) < 10**6*zero_tol and fastMode:
            return 0
        else:
            raise ValueError('Invalid value encountered in Bures distance.')
    
    def faster_RootBuresFidelity(A, B, *args, **kwargs):
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else linalg.fractional_matrix_power(A, 1/2)
        Bhalf = kwargs['Bhalf'] if 'Bhalf' in kwargs.keys() else linalg.fractional_matrix_power(B, 1/2)
        fastMode = kwargs['fastMode'] if 'fastMode' in kwargs.keys() else False
        mat = Ahalf@Bhalf
        if not fastMode:
            val = torch.sum(torch.linalg.svdvals(Ahalf@Bhalf))
        else:
            val = torch.sum(torch.abs(torch.linalg.eigvalsh(mat@mat.T))**(1/2)) # Even faster, but has some precision problems
            val += torch.sum(torch.abs(torch.linalg.eigvalsh(mat.T@mat))**(1/2)) # Average with other direction to hope for better accuracy
            val = val / 2
        return val
        
    def BuresAngle(A, B, *args, **kwargs):
        """Only applicable to PSD matrices A, B with trace = 1, so that RootBuresFidelity is bounded in [-1,1]"""
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else linalg.fractional_matrix_power(A, 1/2)
        Bhalf = kwargs['Bhalf'] if 'Bhalf' in kwargs.keys() else linalg.fractional_matrix_power(B, 1/2)
        fastMode = kwargs['fastMode'] if 'fastMode' in kwargs.keys() else False
        val = distances.faster_RootBuresFidelity(A, B, Ahalf=Ahalf, Bhalf=Bhalf, fastMode=fastMode)
        return torch.arccos(val).real

    def AffineInvariant(A, B, *args, **kwargs):
        "SPD, not SPSD"
        Ahalf = kwargs['Ahalf'] if 'Ahalf' in kwargs.keys() else linalg.fractional_matrix_power(A, 1/2)
        A_neghalf = kwargs['A_neghalf'] if 'A_neghalf' in kwargs.keys() else torch.linalg.inv(Ahalf)
        temp = A_neghalf @ B @ A_neghalf
        # TODO: Compare speed of SVD log on GPU with scipy log on cpu (which is worse, SVD or GPU/CPU back and forth?)
        matLog = linalg.matrixLog(temp)
        return torch.linalg.norm(matLog)
    
    def LogFrobenius(A, B, *args, **kwargs):
        "SPD, not SPSD"
        logA = linalg.matrixLog(A)
        logB = linalg.matrixLog(B)
        return torch.linalg.norm(logA - logB)
    
    def Euclidean(A, B, *args, **kwargs):
        return torch.linalg.norm(A - B)
    
    # def BuresDistance(A, B, Ahalf=None, A_neghalf=None, featureDist=None, zero_tol=10**-10):
    #     """Recommended: Compute A^(1/2) (i.e. Ahalf) and pass that into the method. This will be faster for pairwise distance loops.
    #     For more, see Rajendra Bhatia, Tanvi Jain, Yongdo Lim.
    #     Paper Title: On the Bures-Wasserstein distance between positive definite matrices"""
    #     # val = torch.trace(A) + torch.trace(B) - 2 * torch.trace(linalg.fractional_mat_power(Ahalf@B@Ahalf, 1/2))
    #     val = torch.trace(A) + torch.trace(B) - 2 * distances.RootBuresFidelity(A,B,Ahalf,A_neghalf,featureDist)
    #     if torch.abs(val) > zero_tol:
    #         return torch.sqrt(val.to(torch.complex128)).real # NOTE: Switching order of A and B inputs appears to only meaningfully affect imaginary part.
    #     elif torch.abs(val) <= zero_tol:
    #         return 0
    
    # def RootBuresFidelity(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
    #     if Ahalf is None:
    #         Ahalf = linalg.fractional_mat_power(A, 1/2)
    #     val = torch.trace(linalg.fractional_mat_power(Ahalf@(B.to(torch.complex128))@Ahalf, 1/2))
    #     # val = torch.trace(linalg.fractional_matrix_power(B@A, 1/2))
    #     return val
        
    # def BuresAngle(A, B, Ahalf=None, A_neghalf=None, featureDist=None):
    #     # Only applicable to normalized matrices?
    #     # Need the %2 to put numbers in range of arccos; not sure if that's the best way to handle it
    #     # val = (np.sqrt(Distances.BuresFidelity(A, B, Ahalf, A_neghalf)) + 1) % 2 - 1
    #     val = min(1,max(0, distances.RootBuresFidelity(A, B, Ahalf, A_neghalf).real ))
    #     return torch.arccos(val).real