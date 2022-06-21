from typing import Tuple
import numpy as np


from scipy.optimize import fmin_l_bfgs_b

from src_lib import *

class Kernel:
    def __init__(self, kname: str = "Mat", d: int=2, c: float=0.0, gamma: float = 1.0):
        self.kname = kname
        self.d = d
        self.c = c
        self.gamma = gamma

    def _dotPoly2(self, x1:np.ndarray, x2: np.ndarray, d: int, c: float) -> np.ndarray:
        x1T = x1.T

        m = np.dot(x1T, x2) + c
        return np.power(m, d)

    def _dotRBF(self, x1:np.ndarray, x2: np.ndarray, gamma: float)-> np.ndarray:
        X1_norm = np.sum(x1 **2, axis = 0)
        X2_norm = np.sum(x2 **2, axis=0)
        
        #exploiting: ||x-y||^2 = ||x||^2 + ||y||^2 -2*x.T * y
        ret = np.exp(-gamma*(X1_norm[:, None] +X2_norm[None,:] - 2*np.dot(x1.T,x2)))


        return ret

    def _dotMat(self, x1: np.ndarray, x2:np.ndarray)-> np.ndarray:
        return np.dot(x1.T, x2)

    def dot(self, x1: np.ndarray, x2:np.ndarray):
        if self.kname == "poly2":
            return self._dotPoly2(x1,x2, self.d, self.c)
        elif self.kname == "RBF":
            return self._dotRBF(x1,x2, self.gamma)
        else:
            return self._dotRBF(x1,x2, self.gamma)

class SVMNL_Model(Model):
    def __init__(self, n_classes: int, K: float, C: float, kernel: Kernel = Kernel(), preProcess: PreProcess = PreProcess("None"), rebalance: bool = False):
        super().__init__(n_classes, preProcess=preProcess)

        self.C = C
        self.kernel = kernel
        self.K = K
        self.rebalance= rebalance
        

    def _compute_L(self, alpha: np.ndarray) -> Tuple[float, np.ndarray]:
        a = vcol(alpha)
        L = 0.5*np.dot(np.dot(a.T, self.H), a)- np.dot(a.T, vcol(np.ones((a.shape[0]))))

        Lgrad = (np.dot(self.H, a) - vcol(np.ones((a.shape[0])))).reshape((a.shape[0]))
        
        return (L, Lgrad)

    def train(self, D: np.ndarray, L: np.ndarray):
        D, L = self.preProcess.learn(D, L)
        self.DTR = D
        G = self.kernel.dot(D, D) + self.K

        Lz = np.ones((L.shape[0]))
        Lz[L == 0] = -1
        self.Lz = Lz
        Z= np.dot(vcol(Lz),vrow(Lz))

        self.H = Z*G

        x0 = np.zeros((D.shape[1]))
        if self.rebalance== False:
            boundsList = [(0,self.C)] *D.shape[1]
        else:
            Ct = (self.C*self.prior[1])/(L==1).mean()
            Cf = (self.C*self.prior[1])/(L==0).mean()
            boundsList = [(0, Ct if label == 1 else Cf) for label in L]
        
        x,f,d=fmin_l_bfgs_b(self._compute_L, x0, iprint=0, bounds= boundsList, factr= 1.0, maxiter = 100000, maxfun = 100000)
        self.alphas = x


    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D, L = self.preProcess.apply(D, L)
        G = self.kernel.dot(self.DTR, D) + self.K
        
        AZ= self.Lz*self.alphas
        AZG =  G * vcol(AZ)
        

        S = np.sum(AZG, axis = 0)
        
        preds = S>0
        true = L >0
        acc=np.sum(preds==true)/L.shape[0]

        return acc, preds, S

    
class SVML_Model(Model):
    def __init__(self, n_classes: int, K: float, C: float, preProcess: PreProcess = PreProcess("None"), rebalance: bool = False):
        super().__init__(n_classes, preProcess=preProcess)
        self.K = K
        self.C = C
        self.rebalance = rebalance

    def _build_vecs(self, D: np.ndarray, to_add: float) -> np.ndarray:
        m = np.ones((D.shape[0]+1, D.shape[1]))
        m[0:D.shape[0], 0:D.shape[1]] = D
        m[D.shape[0], :] = to_add
        return m

    def _compute_L(self, alpha: np.ndarray) -> Tuple[float, np.ndarray]:
        a = vcol(alpha)
        L = 0.5*np.dot(np.dot(a.T, self.H), a)- np.dot(a.T, vcol(np.ones((a.shape[0]))))

        Lgrad = (np.dot(self.H, a) - vcol(np.ones((a.shape[0])))).reshape((a.shape[0]))
        
        return (L, Lgrad)
    

    def train(self, D: np.ndarray, L: np.ndarray):
        D, L = self.preProcess.learn(D, L)
        Dc = self._build_vecs(D, self.K)
    

        G = np.dot(Dc.T, Dc)

        Lz = np.ones((L.shape[0]))
        Lz[L == 0] = -1
        Z= np.dot(vcol(Lz),vrow(Lz))

        self.H = Z*G
        
        x0 = np.zeros((Dc.shape[1]))
        if self.rebalance== False:
            boundsList = [(0,self.C)] *Dc.shape[1]
        else:
            Ct = (self.C*self.prior[1])/(L==1).mean()
            Cf = (self.C*self.prior[1])/(L==0).mean()
            boundsList = [(0, Ct if label == 1 else Cf) for label in L]
        #print(boundsList)
        
        x,f,d=fmin_l_bfgs_b(self._compute_L, x0, iprint=0, bounds= boundsList, factr= 1.0, maxiter = 100000, maxfun = 100000)
        self.alphas = x

        Dcz = Dc*Lz
        self.w = np.dot(Dcz, vcol(self.alphas))
        
    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D, L = self.preProcess.apply(D, L)

        Dc = self._build_vecs(D, self.K)

        S = np.dot(vrow(self.w), Dc)

        
        preds = S>0
        true = L >0
        acc=np.sum(preds==true)/L.shape[0]
        
        return acc, preds, np.reshape(S, (S.shape[1]))