from re import S
import numpy as np
import itertools as itt
import matplotlib.pyplot as plt
from typing import Tuple

from src_lib.utils import *

from scipy.stats import norm, rankdata

"""
    Classes and methods meant to apply preprocessing of data
"""

class PreProcess:
    def __init__(self, name: str):
        self.name = name
        self.next: PreProcess = None
    
    def learn(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return D, L
    
    def apply(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return D, L
    
    def addNext(self, next) -> None:
        if self.next is None:
            self.next = next
        else:
            self.next.addNext(next)


class PCA(PreProcess):
    def __init__(self, m: int):
        super().__init__("PCA")
        self. m = m
    
    def _PCA_compute(self, D:np.ndarray)->np.ndarray:
        return np.dot(self.P.T, D)

    def learn(self, D: np.ndarray, L:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        C = get_covarianceCentered(D)
        s, U = np.linalg.eigh(C)

        self.P = U[:, ::-1][:, 0:self.m]

        ret = self._PCA_compute(D)

        if self.next is None:
            return ret, L
        else:
            return self.next.learn(ret, L)

    def apply(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.next is None:
            return self._PCA_compute(D), L
        else:
            return self.next.apply(self._PCA_compute, L)

class LDA(PreProcess):
    def __init__(self, m: int ):
        super().__init__("LDA")
        self.m = m

    def _LDA_compute(self, D:np.ndarray) -> np.ndarray:
        return np.dot(self.W.T, D)


    def learn(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        SB= get_betweenCovariance(D, L)
        SW = get_withinCovariance(D, L)


        self.W = joint_diagonalization(SB, SW, self.m)

        if self.next is None:
            return self._LDA_compute(D), L
        else:
            return self.next.learn(self._LDA_compute(D), L) 

    def apply(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.next is None:
            return self._LDA_compute(D), L
        else:
            return self.next.apply(self._LDA_compute, L)

class Gaussianize(PreProcess):
    def __init__(self ):
        super().__init__("gauss")
    
    def _Gauss_compute(self, D: np.ndarray) -> np.ndarray:
        DG = np.zeros((D.shape[0], D.shape[1]))
        
        for i in range(0, D.shape[1]):
            
            DG[:, i] = np.argmax(self.valMat > vcol(D[:, i]), axis=1)

        DG += 1
        DG= np.clip(DG/(D.shape[1]+2), 0, 0.9999)    
        DG = norm.ppf(DG)
        


        return DG


    def learn(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        rank_i = np.argsort(D, axis=1) 

        self.valMat = np.zeros((D.shape[0], D.shape[1]))

        for i in range(0, D.shape[0]):
            self.valMat[i, :] = D[i, rank_i[i]]

        
        if self.next is None:
            return self._Gauss_compute(D), L
        else:
            return self.next.learn(self._Gauss_compute(D), L) 

        
    def apply(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.next is None:
            return self._Gauss_compute(D), L
        else:
            return self.next.apply(self._Gauss_compute, L)
        
