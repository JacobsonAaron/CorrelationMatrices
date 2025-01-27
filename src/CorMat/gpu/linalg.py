import torch

class linalg():
    
    def fractional_mat_power(A, p):
        if isinstance(A, torch.Tensor):
            # Not as sophisticated as the scipy method, but it supports GPU computation
            evals, evecs = torch.linalg.eig(A)
            evals = evals
            evpow = evals**(p)
            return torch.matmul(evecs, torch.matmul (torch.diag(evpow), torch.inverse(evecs)))
        else:
            raise TypeError(f"Unsupported type found in fractional_mat_power: {type(A)}")
        
    def matrixLog(mat):
        # TODO: implement this
        raise NotImplementedError