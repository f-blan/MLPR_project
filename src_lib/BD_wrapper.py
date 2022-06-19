
import numpy as np
from src_lib.Model import *
from src_lib.MVG_Model import *
from src_lib.Discrete_Model import *
from typing import Tuple, List
import matplotlib.pyplot as plt

class BD_Wrapper:
    def __init__(self, modelType: str, n_classes: int, C: np.ndarray, e_prior: float, model: Model = None):
        
        if model is None:
            if(modelType=="mvg"):
                self.model = MVG_Model(n_classes, False,False, prior=np.array([1-e_prior, e_prior]))
            elif(modelType=="mvg_tied"):
                self.model = MVG_Model(n_classes,True,False, prior=np.array([1-e_prior, e_prior]))
            elif(modelType=="mvg_naive"):
                self.model =MVG_Model(n_classes, False,True, prior=np.array([1-e_prior, e_prior]))
            elif(modelType=="mvg_tied_naive"):
                self.model =MVG_Model(n_classes, True,True, np.array([1-e_prior, e_prior]))
            elif(modelType=="discrete"):
                DTR, DTE, LTE, _ = get_Inf_Par()
                self.model =Discrete_Model(n_classes, 0.001, prior = np.array([1-e_prior, e_prior]), label_translate=_)
        else:
            self.model = model

        self.n_classes = n_classes
        self.C = C
        self.prior = np.array([1-e_prior, e_prior])

    
    def train(self, D:np.ndarray, L: np.ndarray) -> None:
        self.model.train(D, L)
    
    def predict(self, D: np.ndarray, L:np.ndarray, th: float=None) -> Tuple[float, np.ndarray, np.ndarray]:
        _, __, S =   self.model.predict(D,L)
        
        
        llrs = np.log(S[1, :]/S[0, :])
        if th == None:
            th = -np.log((self.prior[1]*self.C[0, 1])/(self.prior[0]*self.C[1,0]))
        preds = llrs
        i1 = llrs>th
        i2 = llrs<=th
        preds[i1] = 1
        preds[i2] = 0
        acc = np.sum(preds == L)/L.shape[0]
        return acc, preds, S

    
    def computeConfusionMatrix(self, D:np.ndarray, L:np.ndarray) -> np.ndarray:
        _, predL, __= self.predict(D,L)
        labels = L
        m = np.zeros((self.n_classes, self.n_classes))
        for i in range(L.shape[0]):
            labelx = int(labels[i])
            labely = int(predL[i])
            m[labely, labelx] += 1
        
        print(f"acc: {_}")

        return m

    def get_risk(self, M: np.ndarray) -> float:
        FNR = M[0,1]/(M[0,1]+M[1,1])
        FPR = M[1,0]/(M[1,0]+M[0,0])

        return self.prior[1]*FNR*self.C[0,1]+self.prior[0]*FPR*self.C[1,0]

    def get_norm_risk(self, M: np.ndarray) -> float:
        risk = self.get_risk(M)
        opt_risk = np.min(np.array([self.prior[1]*self.C[0,1], self.prior[0]*self.C[1,0]]))
        return risk/opt_risk

    def get_matrix_from_threshold(self, L:np.ndarray, llrs:np.ndarray, th: float) -> np.ndarray:
        predL = np.int32(llrs > th)
        labels = L
        m = np.zeros((self.n_classes, self.n_classes))
        for i in range(0, L.shape[0]):
            labelx = int(labels[i])
            labely = int(predL[i])
            m[labely, labelx] += 1
        return m


    def compute_best_threshold(self, D:np.ndarray, L:np.ndarray) -> Tuple[float, float]:
        _, __, Post =   self.model.predict(D,L)
        labels = L
        
        llrs = np.log(Post[1, :]/Post[0, :])
        thresholds = np.copy(llrs)
        thresholds.sort()

        thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
        risks=[]
        best_i = 0

        for idx, t in enumerate(thresholds):
            m = self.get_matrix_from_threshold(labels,llrs,t)
            r = self.get_norm_risk(m)
            risks.append(r)
            if r<=risks[best_i]:
                best_i = idx

        return min(risks), thresholds[best_i]

    def plot_ROC_over_thresholds(self, D: np.ndarray, L:np.ndarray) -> None:
        _, __, Post =   self.model.predict(D,L)
        labels = L
        
        llrs = np.log(Post[1, :]/Post[0, :])
        thresholds = np.copy(llrs)
        thresholds.sort()

        thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
        TPRs = []
        FNRs = []
        for idx, t in enumerate(thresholds):
            m = self.get_matrix_from_threshold(labels,llrs,t)
            FNR = m[0,1]/(m[0,1]+m[1,1])
            FPR = m[1,0]/(m[1,0]+m[0,0])
            TPR = 1-FPR

            TPRs.append(TPR)
            FNRs.append(FNR) 

        plt.plot(FNRs, TPRs )
        plt.show()

    def plot_Bayes_errors(self, D, L) -> None:
        _, __, Post =   self.model.predict(D,L)
        labels = L
        
        llrs = np.log(Post[1, :]/Post[0, :])
        old_prior = self.prior
        priorLogOdds = np.linspace(-3, 3, 21)
        DCFs = []
        minDCFs = [] 
        for idx, p in enumerate(priorLogOdds):
            effectiveP = 1/(1+np.exp(-p))
            th = -np.log((effectiveP)/(1-effectiveP))
            
            self.prior = np.array([(1-effectiveP), effectiveP])
            m = self.get_matrix_from_threshold(labels, llrs, th=th)
            

            DCFs.append(self.get_norm_risk(m))
            minDCFs.append(self.compute_best_threshold(D,L)[0])

        self.prior = old_prior        

        DCFs.reverse()
        
        minDCFs.reverse()
        plt.plot(priorLogOdds, DCFs, label='DCF', color= 'r')

        plt.plot(priorLogOdds, minDCFs, label='minDCF', color= 'b')
        plt.ylim([0, 1.1])
        plt.xlim([-3,3])
        plt.show()

    def checker(self, th: float = None) -> Tuple[float, np.ndarray]:
        if th == None:
            th = -np.log((self.prior[1]*self.C[0, 1])/(self.prior[0]*self.C[1,0]))
        #print(llrs)
        #print(llrs.shape)
        llrs = np.load('commedia_llr_infpar.npy')
        labels = np.load('commedia_labels_infpar.npy')
        print(f"th: {th}")
        
        preds = llrs
        i1 = llrs>th
        i2 = llrs<=th
        #print((i1==i2).sum())
        #print((llrs<=th).sum())
        preds[i1] = 1
        preds[i2] = 0
        acc = np.sum(preds == L)/L.shape[0]
        return acc, preds
