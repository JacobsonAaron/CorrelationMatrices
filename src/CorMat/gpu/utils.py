import os
import torch
from scipy import linalg as la
from typing import Callable, Iterable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class utils():

    def calculatePairwiseDistances(Matrices: Iterable, distance: Callable, Mats_half: Iterable = None, Mats_neghalf: Iterable = None, featureDist=None, 
                            computeHalves: bool = False, computeNeghalves: bool = False, 
                            assumeDistIsSymmetric: bool = False, silent: bool = False):
        """Calculates a matrix of pairwise distances between objects in an iterable."""
        # TODO: Allow this to resume progress if interrupted?
        if Mats_half is None and computeHalves == True:
            raise ValueError('Mats_half cannot be None if computeNeghalves is True. Alternatively, use CPU version of this function.')
        if Mats_neghalf is None and computeNeghalves == True:
            raise ValueError('Mats_neghalf cannot be None if computeNeghalves is True. Alternatively, use CPU version of this function.')
        
        numArrays = len(Matrices)
        pairwiseDists = torch.zeros((numArrays, numArrays), dtype=float).to(device)
        j=0
        for i in range(numArrays):
            innerLoopUpperIdx = i+1 if assumeDistIsSymmetric else numArrays
            if i%10 == 0 and not silent:
                print("i =", i, "/", numArrays, "|", "j =", j, "          ", end="\r")
            A = Matrices[i]
            Ahalf = Mats_half[i] if Mats_half is not None else None
            A_neghalf = Mats_neghalf[i] if Mats_neghalf is not None else None
            for j in range(0,innerLoopUpperIdx):
                if j%30 == 0 and not silent:
                    print("i =", i, "/", numArrays, "|", "j =", j, "          ", end="\r")
                B = Matrices[j]
                dist = distance(A, B, Ahalf, A_neghalf, featureDist)
                pairwiseDists[i,j] = dist
                if assumeDistIsSymmetric:
                    pairwiseDists[j,i] = dist
        return pairwiseDists