import numpy as np
import scipy as sp
from typing import Tuple, List, Optional

from src_lib.LR_Model import LRBinary_Model
from src_lib.Model import Model
from src_lib.PreProcess import PreProcess


from src_lib.utils import *

from threading import Thread
from typing import Optional, Tuple

#this class is meant to work mainly
class Fusion_Model(Model):
    def __init__(self, n_classes: int, kcv_obj, preProcess: PreProcess = PreProcess("None"),verbose: bool = False, calibrate_after: bool = True, e_prior: float = 0.5):
        super().__init__(n_classes, preProcess=preProcess)
        self.models : List[Model] = []
        self.calibrators: List[Optional[Model]] = []
        self.calibrate_after = calibrate_after #calibrate each model individually or calibrate the combination of all model scores
        self.e_prior = e_prior
        self.kcv_obj = kcv_obj
        self.kcv_obj.model = self
        self.mode = "crossValidate"
        if self.calibrate_after:
            self.all_calibrator = LRBinary_Model(2, 0, rebalance=True, prior = np.array([1-e_prior, e_prior]))
    
    def add_model(self, model: Model, calibrate: bool, e_prior: float = 0.5) -> None:
        #this can be used also to add a pretrained model (use replace_calibrator to add a pretrained calibrator)
        self.models.append(model)
        self.calibrators.append(LRBinary_Model(2, 0, rebalance=True, prior = np.array([1-e_prior, e_prior])) if calibrate else None)
    
    def replace_calibrator(self, calibrator: Model, i: int) -> None:
        #this function is here just to allow using a pretrained calibrator during decision evaluation
        self.calibrators[i] = calibrator

    def train_calibrator(self, D: np.ndarray, L: np.ndarray, mode: str)->Optional[Tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray,np.ndarray]] :
        prev_mode = self.mode
        self.mode = "crossVal"
        _, scores = self.kcv_obj.crossValidateFusion(D, L)
        self.mode = prev_mode

        if mode == "crossVal":
            (t_S, t_L), (v_S, v_L) = shuffle_and_split_dataset(scores, L, dims = 2)
            self.all_calibrator.train(t_S, t_L)
            return t_S, t_L, v_S, v_L, scores
        else:
            self.all_calibrator.train(scores, L)
        



    def _train_ith_model(self, D: np.ndarray, L:np.ndarray, i: int):
        if self.calibrators[i] is None or self.calibrate_after:
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

        if self.calibrators[i] is not None and self.calibrate_after == False:
            _, __, scores = self.calibrators[i].predict(scores, L)
        
        self.predictedScores[i, :] = vrow(scores)

    
    def train(self, D: np.ndarray, L: np.ndarray):
        D, L = self.preProcess.learn(D, L) #learn root preprocessing if any (in general each model does its own preprocessing in its train method)
        threadPool: List[Thread] = []

        if self.mode != "crossVal":
            self.train_calibrator(D, L, "fullTrain")

        for i in range(0, len(self.models)):
            th = Thread(target=self._train_ith_model, args=(D, L, i))
            th.start()
            threadPool.append(th)
        
        for th in threadPool:
            th.join()
        #if self.calibrate_after is True it's needed to train the calibrator separately on the result of k-fold on the train set


    def predict(self, D: np.ndarray, L: np.ndarray):
        D, L =self.preProcess.learn(D, L)
        #WARNING: this function doesn't also return actual predictions like Model does. Scores should be used with a BD_Wrapper 
        self.predictedScores = np.zeros((len(self.models), L.shape[0]))

        threadPool: List[Thread] = []

        for i in range(0, len(self.models)):
            th = Thread(target=self._predict_ith_model, args=(D, L, i))
            th.start()
            threadPool.append(th)
        
        for th in threadPool:
            th.join()

        if self.calibrate_after and self.mode != "crossVal":
            self.predictedScores = self.all_calibrator.predict(self.predictedScores, L)
            self.predictedScores.sum(axis = 0)
        
        #fuse models
        S = self.predictedScores

        #dummy accuracy metrics
        predL = S>0
        acc = compute_acc(predL, L)

        return acc, predL, S




