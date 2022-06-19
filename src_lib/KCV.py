from src_lib.Model import Model
import numpy as np
from typing import List, Tuple
import math

class KCV:
    def __init__(self, model: Model, K: int, LOO: bool = False):
        self.model = model
        self.K = K
        self.LOO = LOO

    def _split_into_k(self, D: np.ndarray, L: np.ndarray, K: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        parts: List[Tuple[np.ndarray, np.ndarray]] = []

        split_f = math.floor(D.shape[1]/K)

    
        i = 0
        while i < D.shape[1]-split_f:
            parts.append((D[:, i:i+split_f], L[i:i+split_f]))
            i+=split_f
        parts.append((D[:, i:], L[i:]))
        return parts

    def _create_partition(self, parts: List[Tuple[np.ndarray, np.ndarray]], held_out: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,np.ndarray]]:
        size = 0

        for i,part in enumerate(parts):
            if i != held_out:
                size += part[0].shape[1]
    
    
        DTR = np.zeros((parts[0][0].shape[0], size))
        LTR = np.zeros(size, dtype=int)
        index = 0

        for i, part in enumerate(parts):
            if i!=held_out:
                batch_size = part[0].shape[1]
                DTR[:, index:index+batch_size] = part[0]
            
                LTR[index:index+batch_size] = part[1]
                index+=batch_size
    
    
        return (DTR, LTR), (parts[held_out][0], parts[held_out][1])
        

    def crossValidate(self, D: np.ndarray, L: np.ndarray) -> float:
        K = self.K

        if self.LOO:
            K = L.shape[0]

        
        parts = self._split_into_k(D, L, K)
        tot_c = 0
        tot_s = 0

        for held_out in range(0, K):
            (DTR, LTR), (DTE,LTE) = self._create_partition(parts, held_out)

            self.model.train(DTR, LTR)

            _, preds, __ = self.model.predict(DTE, LTE)

            tot_c += (preds==LTE).sum()
            tot_s += LTE.shape[0]
        
        return tot_c/tot_s
    

    def find_best_par(self ):
        pass
