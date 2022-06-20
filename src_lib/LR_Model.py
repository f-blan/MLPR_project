import numpy as np

from src_lib.Model import Model
from src_lib.PreProcess import PreProcess
from src_lib.utils import *

from typing import Tuple

from scipy.optimize import fmin_l_bfgs_b

class LRBinary_Model(Model):
    def __init__(self, n_classes: int,  reg_lambda:float, preProcess: PreProcess = PreProcess("None"), rebalance = False):
        super().__init__(n_classes, preProcess=preProcess)
        self.reg_lambda = reg_lambda
    
    def _logRegFun(self, v: np.ndarray) -> float:
        w, b = v[0:-1], v[-1]
        wt= vcol(w).T
        
        
        const = (self.reg_lambda/2)*np.power(np.linalg.norm(w),2)

        S = np.dot(wt, self.DTR)+b
        cxe=np.logaddexp(0, -S*self.z).mean()
        return const + cxe 


    def train(self, D: np.ndarray, L: np.ndarray) :
        D,L = self.preProcess.learn(D, L)
        self.DTR = D
        self.LTR = L

        self.z = L.copy()
        self.z[L==0]=-1
        
        x0 = np.zeros((D.shape[0]+1))
        x,f,d=fmin_l_bfgs_b(self._logRegFun, x0, approx_grad = True, iprint=0)

        self.w, self.b = x[0:-1], x[-1] 

    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D, L = self.preProcess.apply(D, L)
        wt= vcol(self.w).T
        S=np.dot(wt, D) +self.b
        preds = S>0
        true = L >0
        acc=np.sum(preds==true)/preds.shape[1]

        return acc, preds, np.reshape(S, (S.shape[1]))