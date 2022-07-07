from statistics import mode
from src_lib.BD_wrapper import BD_Wrapper
from src_lib.LR_Model import LRBinary_Model
from src_lib.MVG_Model import MVG_Model
from src_lib.SVM_Model import SVMNL_Model, SVML_Model

from src_lib.utils import *
from src_lib.Model import Model
import numpy as np
from typing import Any, List, Tuple
import math

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

    

    def find_best_par(self, model: Model, D: np.ndarray, L:np.ndarray, par_index: int, bounds: Tuple[float, float], logBounds: bool = True, logbase: float=10.0, e_prior: float = 0.5, verbose:bool = False, n_vals: int=20) -> Any:
        
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
        elif type(model).__name__ == "GMMLBG_Model" or type(model).__name__ == "GMMLBG_Diag_Model" or type(model).__name__ == "GMMLBG_Tied_Model":
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

            print(f"minDCF: {minDCF}, acc: {acc}")
            minDCFs.append(minDCF)
            
        
        return minDCFs, par_vals, accs
        
        
