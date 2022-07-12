from statistics import mode
from src_lib.BD_wrapper import BD_Wrapper
from src_lib.C_Wrapper import C_Wrapper
from src_lib.Fusion_Model import Fusion_Model
from src_lib.LR_Model import LRBinary_Model
from src_lib.MVG_Model import MVG_Model
from src_lib.SVM_Model import SVMNL_Model, SVML_Model

from src_lib.utils import *
from src_lib.Model import Model
import numpy as np
from typing import Any, List, Tuple
import math
import random as r

VERBOSE=False

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
        
    def crossValidateFusion(self, D: np.ndarray, L: np.ndarray, verbose:bool = False) -> float:
        K = self.K

        if self.LOO:
            K = L.shape[0]

        
        parts = self._split_into_k(D, L, K)
        tot_c = 0
        tot_s = 0

        whole_S = np.zeros((len(self.model.models), D.shape[1]))
        whole_S_i = 0

        for held_out in range(0, K):
            (DTR, LTR), (DTE,LTE) = self._create_partition(parts, held_out)

            self.model.train(DTR, LTR)

            _, preds, S = self.model.predict(DTE, LTE)

            
            whole_S[:, whole_S_i : whole_S_i + S.shape[1]] = S
            whole_S_i+=S.shape[1]
            tot_c += (preds==LTE).sum()
            tot_s += LTE.shape[0]
        
        
        return tot_c/tot_s, whole_S
    
    def crossValidate(self, D: np.ndarray, L: np.ndarray, verbose:bool = False) -> float:
        K = self.K

        if self.LOO:
            K = L.shape[0]

        
        parts = self._split_into_k(D, L, K)
        tot_c = 0
        tot_s = 0

        whole_S = np.zeros((D.shape[1]))
        whole_S_i = 0

        for held_out in range(0, K):
            (DTR, LTR), (DTE,LTE) = self._create_partition(parts, held_out)

            self.model.train(DTR, LTR)

            _, preds, S = self.model.predict(DTE, LTE)

            whole_S[whole_S_i : whole_S_i + S.shape[0]] = S
            whole_S_i+=S.shape[0]
            tot_c += (preds==LTE).sum()
            tot_s += LTE.shape[0]
        
        return tot_c/tot_s, whole_S

    
    def compute_min_dcf(self, model: Model, D: np.ndarray, L:np.ndarray, e_prior: float = 0.5):
        w=BD_Wrapper("Static", 2, e_prior=e_prior, model=model)
        self.model = model
        acc, whole_S = self.crossValidate(D, L, VERBOSE)

        minDCF, th = w.compute_best_threshold_from_Scores(whole_S, L)

        return minDCF, th

    def compute_actual_dcf(self, model: Model, D: np.ndarray, L: np.ndarray, e_prior: float = 0.5, th: float = None):
        w=BD_Wrapper("Static", 2, e_prior=e_prior, model=model)
        self.model = model
        acc, whole_S = self.crossValidate(D, L, VERBOSE)
        if th is None:
            th = w.get_theoretical_threshold()
        confusion_m = w.get_matrix_from_threshold(L, whole_S, th)
        
        actDcf = w.get_norm_risk(confusion_m)

        return actDcf, th
    
    def compute_min_actual_dcf_fusion(self, model: Fusion_Model, D: np.ndarray, L: np.ndarray, e_prior: float = 0.5, th: float = None):
        #thematically closer with calibrator_eval and threshold_estimate (they use the same train and evaluation sets)
        
        self.model = model
        model.mode = "crossVal"

        t_S, t_L, v_S, v_L, S = model.train_calibrator(D, L, "crossVal")
        w=BD_Wrapper("Static", 2, e_prior=e_prior, model=model)

        _,_,v_S = model.all_calibrator.predict(v_S, v_L) 
        minDCF,best_th = w.compute_best_threshold_from_Scores(v_S, v_L)

        theory_th = w.get_theoretical_threshold()
        confusion_m = w.get_matrix_from_threshold(v_L, v_S, theory_th)
        actDCF = w.get_norm_risk(confusion_m)

        return minDCF, actDCF, best_th, theory_th

    
    def threshold_estimate(self, model: Model, D: np.ndarray, L: np.ndarray, e_prior: float = 0.5 ):
        w=BD_Wrapper("Static", 2, e_prior=e_prior, model=model)
        self.model = model
        acc, whole_S = self.crossValidate(D, L, VERBOSE)
        (t_S, t_L), (v_S, v_L) = shuffle_and_split_dataset(whole_S, L)

        _, best_th = w.compute_best_threshold_from_Scores(t_S, t_L)
        theory_th = w.get_theoretical_threshold()
        
        minDCF, _ = w.compute_best_threshold_from_Scores(v_S, v_L)

        confusion_m = w.get_matrix_from_threshold(v_L, v_S, theory_th)
        theory_actDCF = w.get_norm_risk(confusion_m)

        confusion_m = w.get_matrix_from_threshold(v_L, v_S, best_th)
        estimate_th_actDCF = w.get_norm_risk(confusion_m)

        return minDCF, theory_actDCF, estimate_th_actDCF, best_th

    def calibrator_eval(self, model: Model, D:np.ndarray, L: np.ndarray, e_prior: float):
        self.model = model
        acc, whole_S = self.crossValidate(D, L, VERBOSE)
        (t_S, t_L), (v_S, v_L) = shuffle_and_split_dataset(whole_S, L)

        cal_w = C_Wrapper(e_prior = e_prior)
        calibrator, _ = cal_w.train(t_S, t_L, eval_mode = True)
        _, __, cal_Scores_eval= calibrator.predict(v_S, v_L)

        w=BD_Wrapper("Static", 2, e_prior=0.5, model=model)
        theory_th = w.get_theoretical_threshold()
        confusion_m = w.get_matrix_from_threshold(v_L, cal_Scores_eval, theory_th)
        theory_actDCFb = w.get_norm_risk(confusion_m)

        w=BD_Wrapper("Static", 2, e_prior=0.9, model=model)
        theory_th = w.get_theoretical_threshold()
        confusion_m = w.get_matrix_from_threshold(v_L, cal_Scores_eval, theory_th)
        theory_actDCFf = w.get_norm_risk(confusion_m)

        w=BD_Wrapper("Static", 2, e_prior=0.1, model=model)
        theory_th = w.get_theoretical_threshold()
        confusion_m = w.get_matrix_from_threshold(v_L, cal_Scores_eval, theory_th)
        theory_actDCFm = w.get_norm_risk(confusion_m)

        return theory_actDCFb, theory_actDCFf, theory_actDCFm


    def compute_bayes_pars(self, model: Model, D: np.ndarray, L: np.ndarray):
        w=BD_Wrapper("Static", 2, model=model)
        self.model = model
        acc, whole_S = self.crossValidate(D, L, VERBOSE)

        return w.plot_Bayes_errors_from_scores(whole_S, L, plot= False)

    def compute_calibrated_bayes_pars(self, model: Model, D:np.ndarray, L:np.ndarray):
        w=BD_Wrapper("Static", 2, model=model)
        self.model = model
        acc, whole_S = self.crossValidate(D, L, VERBOSE)

        #local training set and local eval set
        (t_S, t_L), (v_S, v_L) = shuffle_and_split_dataset(whole_S, L)

        #train calibrator with local training set
        cal_w = C_Wrapper(e_prior = 0.5)
        calibrator, _ = cal_w.train(t_S, t_L, eval_mode = True)
        _, __, cal_Scores_eval= calibrator.predict(v_S, v_L)

        #predict calibrated scores for local evaluation set
        _, __, cal_Scores_eval= calibrator.predict(v_S, v_L)

        #get bayes plot for both uncalibrated and calibrated local validation set
        logOdds, calDCFs, minDCFs = w.plot_Bayes_errors_from_scores(cal_Scores_eval, v_L, plot= False)
        _, uncDCFs, __ = w.plot_Bayes_errors_from_scores(v_S, v_L, plot= False)

        return logOdds, calDCFs, uncDCFs, minDCFs



    def find_best_par(self, model: Model, D: np.ndarray, L:np.ndarray, par_index: int, bounds: Tuple[float, float], logBounds: bool = True, logbase: float=10.0, e_prior: float = 0.5, verbose:bool = False, n_vals: int=20) -> Any:
        # I only realised later that i could have programmed this to compute DCFs for the three main applications at once
        # would have been much cleaner and faster to perform experiments
        if(logBounds):
            par_vals = np.logspace(bounds[0], bounds[1], num=n_vals, base=logbase)
        else:
            par_vals = np.linspace(bounds[0], bounds[1], num=n_vals)

        
        def get_next_LR(val_index):
            model.reg_lambda = par_vals[val_index]
        
        def get_next_SVM_C(val_index):
            model.C = par_vals[val_index]


        def get_next_SVM_K(val_index):
            model.K = par_vals[val_index]

        def get_next_GMM_exp(val_index):
            model.n_gauss_exp = math.floor(par_vals[val_index])
            

        def get_next_GMM_alpha(val_index):
            model.alpha = par_vals[val_index]
        
        def get_next_GMM_bound(val_index):
            model.bound = par_vals[val_index]
                 

        if type(model).__name__ == "MVG_Model":
            print("No hyperparams for mvg model, aborting")
            return
        elif type(model).__name__ == "LRBinary_Model" or type(model).__name__ == "QuadLR_Model":
            get_next_par = get_next_LR
        elif type(model).__name__ == "SVMNL_Model" or type(model).__name__ == "SVML_Model":
            get_next_par =  get_next_SVM_C if par_index == 1  else  get_next_SVM_K
        elif type(model).__name__ == "GMMLBG_Model" or type(model).__name__ == "GMMLBG_Diag_Model" or type(model).__name__ == "GMMLBG_Tied_Model" or type(model).__name__ == "GMMLBG_DT_Model":
            get_next_par =  get_next_GMM_exp if par_index == 0 else get_next_GMM_alpha
            if par_index == 2:
                get_next_par = get_next_GMM_bound
        
        minDCFs: List[float] = []
        accs: List[float] = []
        self.model = model
        
        w=BD_Wrapper("Static", 2, e_prior=e_prior, model=model)
        for i in range(0, par_vals.shape[0]):
            if verbose: print(f"iteration: {i}")
            get_next_par(i)
            acc, whole_S = self.crossValidate(D, L, verbose)

            
            accs.append(acc)
            minDCF, _ =w.compute_best_threshold_from_Scores(whole_S, L)

            #print(f"{model.}")

            print(f"minDCF: {minDCF}, acc: {acc}")
            minDCFs.append(minDCF)
            
        
        return minDCFs, par_vals, accs
        
        
