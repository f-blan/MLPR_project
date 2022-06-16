import numpy as np
from src_lib.Model import Model
from src_lib.PreProcess import PreProcess
from src_lib.utils import *
from typing import List, Tuple
from src_lib.MVG_functionals import *

class MVG_Model(Model):
    def __init__(self,n_classes: int, tied: bool, naive: bool, prior: np.ndarray=-np.ones(1), preProcess: PreProcess = PreProcess("None")):
        super().__init__(n_classes, prior = prior, preProcess=preProcess)
        
        self.tied = tied
        self.naive = naive
        self.pars: List[Tuple[np.ndarray, np.ndarray]] = []


    def train(self, D: np.ndarray, L: np.ndarray):
        D, L = self.preProcess.learn(D,L)

        self.pars = []
        if self.tied == False and self.naive == False:
            #print("TRAINING AS UNCONSTRAINED")
            self.pars = get_MLE_Gaussian_parameters(D, L)
        elif self.tied == True:
            #print(f"TRAINING AS TIED AND NAIVE = {self.naive}")
            self.pars = get_MLE_TiedGaussian_parameters(D,L, naive= self.naive)
        else:
            #print("TRAINING AS UNTIED AND NAIVE")
            self.pars = get_MLE_NaiveGaussian_parameters(D,L)
    
    

    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D,L = self.preProcess.apply(D, L)

        return MVG_Log_Predict(D, L, self.prior, self.pars)
        
