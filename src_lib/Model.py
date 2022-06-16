import numpy as np
from typing import Tuple
from src_lib.PreProcess import PreProcess


from src_lib.utils import vcol

class Model:
    def __init__(self,n_classes: int, prior: np.ndarray = -np.ones(1), preProcess: PreProcess = PreProcess("None")):
        if prior[0] == -1:
            self.prior = vcol(np.ones(n_classes)/n_classes)
            
        else:
            self.prior = prior

        self.n_classes = n_classes
        self.preProcess = preProcess
    
    def train(self, D:np.ndarray, L:np.ndarray) -> None:
        pass

    def predict(self, D:np.ndarray, L:np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        return 0.0, np.zeros(1), np.zeros(1)

    def getConfusionMatrix(self, D: np.ndarray, L:np.ndarray) -> np.ndarray:
        _, predL, __ = self.predict(D, L)

        m = np.zeros((self.n_classes, self.n_classes))

        for i in range(L.shape[0]):
            m[predL[i], L[i]] += 1
        
        return m