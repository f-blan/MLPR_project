import numpy as np
import scipy as sp
from typing import Tuple, List, Optional

from src_lib import *

from threading import Thread


#this class is meant to work mainly
class Fusion_Model(Model):
    def __init__(self, n_classes: int, preProcess: PreProcess = PreProcess("None"),verbose: bool = False):
        super().__init__(n_classes, preProcess=preProcess)
        self.models : List[Model] = []
        self.calibrators: List[Optional[Model]] = []
    
    def add_model(self, model: Model, calibrate: bool, e_prior: float = 0.5):
        #this can be used also to add a pretrained model (use replace_calibrator to add a pretrained calibrator)
        self.models.append(model)
        self.calibrators.append(LRBinary_Model(2, 0, rebalance=True, prior = np.array([1-e_prior, e_prior])) if calibrate else None)
    
    def replace_calibrator(self, calibrator: Model, i: int):
        #this function is here just to allow using a pretrained calibrator during decision evaluation
        self.calibrators[i] = calibrator

    def _train_ith_model(self, D: np.ndarray, L:np.ndarray, i: int):
        if self.calibrators[i] is None:
            self.models[i].train(D, L)
        else:
            #we also need to train the calibrator, we do it by kfolding the training set (actual model is later retrained on full train set)
            model = self.models[i]
            kcv = KCV(model, 5)
            calibrator = self.calibrators[i]

            _, scores = kcv.crossValidate(D, L)
            calibrator.train(scores, L)

            #calibrator trained, now train model on full training set
            model.train(D, L)
    
    def _predict_ith_model(self, D: np.ndarray, L: np.ndarray, i: int):
        _, __, scores = self.models[i].predict(D, L)

        if self.calibrators[i] is not None:
            _, __, scores = self.calibrators[i].predict(scores, L)
        
        self.predictedScores[i, :] = vrow(scores)
    
    def train(self, D: np.ndarray, L: np.ndarray):
        threadPool: List[Thread] = []

        for i in range(0, len(self.models)):
            th = Thread(target=self._train_ith_model, args=(D, L, i))
            th.start()
            threadPool.append(th)
        
        for th in threadPool:
            th.join()

    def predict(self, D: np.ndarray, L: np.ndarray):
        #WARNING: this function doesn't also return actual predictions like Model does. Scores should be used with a BD_Wrapper 
        self.predictedScores = np.zeros((len(self.models), L.shape[0]))

        threadPool: List[Thread] = []

        for i in range(0, len(self.models)):
            th = Thread(target=self._predict_ith_model, args=(D, L, i))
            th.start()
            threadPool.append(th)
        
        for th in threadPool:
            th.join()
        
        #fuse models
        S = self.predictedScores.sum(axis=0)

        #dummy accuracy metrics
        predL = S>0
        acc = compute_acc(predL, L)

        return acc, predL, S




