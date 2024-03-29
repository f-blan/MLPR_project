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
DEBUG_ONLY = False

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
        return np.dot(self.P.T, D- self.mean)

    def learn(self, D: np.ndarray, L:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        C = get_covarianceCentered(D)
        s, U = np.linalg.eigh(C)
        self.mean = vcol(D.mean(1))

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
            return self.next.apply(self._PCA_compute(D), L)

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
        if DEBUG_ONLY==False:
            
        
            for i in range(0, D.shape[1]):
                gm = self.valMat > vcol(D[:, i])
                DG[:, i] = np.argmax(gm, axis=1)
                
                #the above code returns rank 0 if the value is greater than any other in the valmat
                DG[ np.any(gm, axis=1)==False, i] = self.valMat.shape[1]+1
                
                

            DG += 1
            DG= np.clip(DG/(self.valMat.shape[1]+2), 0, 0.9999)    
            DG = norm.ppf(DG)
        
        else:
            for f in range(0, D.shape[0]):
                for idx,x in enumerate(D[f,:]):
                    rank = 0
                    for x_i in self.D[f,:]:
                        if(x_i < x):
                            rank += 1
                    ranks = (rank + 1) /(self.D.shape[1] + 2)
                    DG[f][idx] = norm.ppf(ranks)
                    if((norm.ppf(ranks) >=0 or norm.ppf(ranks) <=0) == False):
                        print(D.shape[1])
                        print(rank)
                        print(ranks)
                        assert False == True

        
        return DG


    def learn(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:

        if DEBUG_ONLY == False:
            rank_i = np.argsort(D, axis=1) 

            self.valMat = np.zeros((D.shape[0], D.shape[1]))
            for i in range(0, D.shape[0]):
                self.valMat[i, :] = D[i, rank_i[i]]
        else:
            self.DTRT = np.zeros((D.shape[0], D.shape[1]))
            self.D = D
            for f in range(0,D.shape[0]):
                self.DTRT[f, :] = norm.ppf(rankdata(D[f, :], method="min")/(D.shape[1] + 2)) 
            #print(self.DTRT)
            #assert False == True
        

        if self.next is None:
            return self._Gauss_compute(D), L
        else:
            return self.next.learn(self._Gauss_compute(D), L) 

        
    def apply(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.next is None:
            return self._Gauss_compute(D), L
        else:
            return self.next.apply(self._Gauss_compute(D), L)

class Znorm(PreProcess):
    def __init__(self ):
        super().__init__("znorm")
    
    def _Znorm_compute(self, D: np.ndarray) -> np.ndarray:
        if DEBUG_ONLY:
            #this was written for debug only but was not used in the project
            #print(f"{vcol(D.mean(1))} - {vcol(D.std(1))}")
            Dn = (D- vcol(D.mean(1)))/vcol(D.std(1))
            return Dn
        else:
            Dn = (D- self.mean)/self.std      

            return Dn


    def learn(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        self.mean = vcol(D.mean(axis = 1))
        self.std = vcol(np.std(D, axis=1))

        

        
        if self.next is None:
            return self._Znorm_compute(D), L
        else:
            return self.next.learn(self._Znorm_compute(D), L) 

        
    def apply(self, D: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.next is None:
            return self._Znorm_compute(D), L
        else:
            return self.next.apply(self._Znorm_compute(D), L)
        
