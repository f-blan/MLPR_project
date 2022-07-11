import numpy as np
import scipy as sp
from typing import Tuple, List

from src_lib.LR_Model import LRBinary_Model
from src_lib.Model import Model
from src_lib.utils import vrow

"""
this is a mostly useless class, i had better plans for it but the way i implemented this (at the end of the development stage)
made it unpractical to use it properly
"""

class C_Wrapper:
    def __init__(self, e_prior: float = 0.5):
        self.calibrator = LRBinary_Model(2, 0, rebalance=True, prior = np.array([1-e_prior, e_prior]))

    def train(self, D: np.ndarray, L: np.ndarray, eval_mode: bool = False, kcv_obj = None):
        
        # this function trains on the scores computed through cross validation
        if eval_mode == False:
            #we need to compute the scores by cross validation
            _, S = kcv_obj.crossValidate(D, L)
        else:
            #D is a subset of the scores (train split)
            S = D
        
        self.calibrator.train(vrow(S), L)

        return self.calibrator, S

    def set_model(self, model: Model):
        #we call this function only after calling "train". It is expected that the model passed as parameter has been trained on the whole
        #training set now
        self.model = model
        

    def predict(self, D: np.ndarray, L: np.ndarray):
        _, __, S = self.model.predict(D, L)
        acc, preds, cS = self.calibrator.predict(vrow(S), L)

        return acc, preds, cS