import numpy as np

from src_lib.Model import Model
from src_lib.PreProcess import PreProcess
from src_lib.utils import *

from typing import Tuple

from scipy.optimize import fmin_l_bfgs_b

class LRBinary_Model(Model):
    def __init__(self, n_classes: int,  reg_lambda:float, preProcess: PreProcess = PreProcess("None"), rebalance = False, prior: np.ndarray = -np.ones(1)):
        super().__init__(n_classes, preProcess=preProcess, prior= prior)
        self.reg_lambda = reg_lambda
        self.rebalance = rebalance
    
    def _logRegFun(self, v: np.ndarray) -> float:
        w, b = v[0:-1], v[-1]
        wt= vcol(w).T
        
        
        const = (self.reg_lambda/2)*np.power(np.linalg.norm(w),2)

        if self.rebalance == False:
            S = np.dot(wt, self.DTR)+b
            cxe=np.logaddexp(0, -S*self.z).mean()
            return const + cxe
        else:
            St = np.dot(wt, self.Dt) + b
            Sf = np.dot(wt, self.Df) + b

            cxT = self.prior[1]*np.logaddexp(0, -St).mean()
            cxF = self.prior[0]*np.logaddexp(0, Sf).mean()

            return const + cxT + cxF



    def train(self, D: np.ndarray, L: np.ndarray) :
        D,L = self.preProcess.learn(D, L)
        self.DTR = D
        self.LTR = L

        
        self.z = L.copy()
        self.z[L==0]=-1
        
        self.nT = (L == 1).sum()
        self.nF = (L == 0).sum()
        if self.rebalance:
            self.Dt = D[:,L == 1]
            self.Df = D[:,L == 0]
            #print(f"nT = {self.nT}, nF = {self.nF}, prior= {self.prior}")

        x0 = np.zeros((D.shape[0]+1))
        x,f,d=fmin_l_bfgs_b(self._logRegFun, x0, approx_grad = True, iprint=0)

        self.w, self.b = x[0:-1], x[-1] 

    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D, L = self.preProcess.apply(D, L)
        wt= vcol(self.w).T
        is_one_dim= wt.shape[0] == 1 and wt.shape[1] == 1
        S=np.dot(wt, D) +self.b if is_one_dim == False else wt*D + self.b
        preds = S>0
        true = L >0
        acc=np.sum(preds==true)/preds.shape[1]

        return acc, preds,  np.reshape(S- np.log(self.nT/self.nF), (S.shape[1])) if self.rebalance == False else np.reshape(S- np.log(self.prior[1]/self.prior[0]), (S.shape[1])) 


class QuadLR_Model(LRBinary_Model):
    def __init__(self, n_classes: int, reg_lambda: float, preProcess: PreProcess = PreProcess("None"), rebalance=False, c: float = 0.0):
        super().__init__(n_classes, reg_lambda, preProcess, rebalance)
        self.c = c
    
    def _map_data(self, D: np.ndarray):
        
        Dexp = np.zeros(((D.shape[0]**2) + D.shape[0], D.shape[1]))

        for i in range(0, D.shape[1]):
            vec = vcol(D[:, i])
            Dexp[0: D.shape[0]**2, i] = np.dot(vec, vec.T).T.ravel()
            Dexp[-D.shape[0]:, i] = vec.ravel()
        
        return Dexp
        

    def _logRegFun(self, v: np.ndarray) -> float:
        w, b = v[0:-self.dim], v[-self.dim : ]
        wt= vcol(v).T
        

        assert v.shape[0] == (self.dim**2) + self.dim
        
        
        const = (self.reg_lambda/2)*np.power(np.linalg.norm(w),2)

        if self.rebalance == False:
            S = np.dot(wt, self.DTR)
            cxe=np.logaddexp(0, -S*self.z).mean()
            return const + cxe
        else:
            St = np.dot(wt, self.Dt) + b
            Sf = np.dot(wt, self.Df) + b

            cxT = self.prior[1]*np.logaddexp(0, -St).mean()
            cxF = self.prior[0]*np.logaddexp(0, Sf).mean()

            return const + cxT + cxF

    def train(self, D: np.ndarray, L: np.ndarray):
        D,L = self.preProcess.learn(D, L)
        self.dim = D.shape[0]
        
        D=self._map_data(D)
        self.DTR = D
        
        self.LTR = L

        
        self.z = L.copy()
        self.z[L==0]=-1
        self.nT = (L == 1).sum()
        self.nF = (L == 0).sum()
        if self.rebalance:
            self.Dt = D[:,L == 1]
            self.Df = D[:,L == 0]
            print(f"nT = {self.nT}, nF = {self.nF}, prior= {self.prior}")

        x0 = np.zeros((D.shape[0]))
        x,f,d=fmin_l_bfgs_b(self._logRegFun, x0, approx_grad = True, iprint=0)

        self.w = x 
    
    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D, L = self.preProcess.apply(D, L)
        Dexp= self._map_data(D)
        wt= vcol(self.w).T
        S=np.dot(wt, Dexp) +self.c
        preds = S>0
        true = L >0
        acc=np.sum(preds==true)/preds.shape[1]

        return acc, preds, np.reshape(S - np.log(self.nT/self.nF), (S.shape[1])) #subrtract prior log odds to get llrs