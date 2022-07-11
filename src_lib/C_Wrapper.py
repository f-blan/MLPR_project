import numpy as np
import scipy as sp
from typing import Tuple, List

from src_lib import *

class C_Wrapper:
    def __init__(self, e_prior: float = 0.5):
        self.calibrator = LRBinary_Model(2, 0, rebalance=True, prior = np.array([1-e_prior, e_prior]))

    def train(self, S: np.ndarray, L: np.ndarray):
        #S are the scores computed from the model through cross validation from the whole training set (or a subset of it if we're making decisions)
        self.calibrator.train(S, L)
    
    def set_model(self, model):
        #we call this function only after calling "train". It is expected that the model passed as parameter has been trained on the whole
        #training set now
        self.model = model


    def predict(self, D: np.ndarray, L: np.ndarray):
        _, __, S = self.model.predict(D, L)
        acc, preds, cS = self.calibrator.predict(S, L)

        return acc, preds, cS