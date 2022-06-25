import numpy as np
from typing import Tuple, List

from src_lib.utils import *

import scipy as sp

def get_score_matrix(DTE: np.ndarray, pars: List[Tuple[np.ndarray, np.ndarray]], log_scores = False) -> np.ndarray:
    n_classes = len(pars)
    vecs=[]

    for i in range(0,n_classes):
        mu = pars[i][0]
        C = pars[i][1]

        vec = logpdf_GAU_ND_Opt(DTE, mu, C)

        vecs.append(vec) if log_scores else vecs.append(np.exp(vec))

    
    return np.array(vecs)

def MVG_Log_Predict(
    D: np.ndarray, 
    L: np.ndarray, 
    prior: np.ndarray, 
    pars: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, np.ndarray, np.ndarray]:

    logS= get_score_matrix(D, pars, log_scores=False)
    
    logSjoint= logS+ np.log(prior)  
    logSMarginal= vrow(sp.special.logsumexp(logSjoint))
    logSPost = logSjoint - logSMarginal
    logPredL = np.argmax(logSPost, axis=0)
    acc, corrects = compute_acc(logPredL, L)


    return acc, logPredL, np.log(logS[1, :]/logS[0, :])

def get_MLE_Gaussian_parameters( D: np.ndarray, L:np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    pars : List[Tuple[np.ndarray, np.ndarray]] =[]

    for i in range(0,np.max(L)+1):
        Dc = D[:, L==i]
        mu = vcol(Dc.mean(1))
        cov = get_covarianceCentered(Dc)
        pars.append((mu,cov))
    
    return pars

def get_MLE_TiedGaussian_parameters(D: np.ndarray, L: np.ndarray, naive: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    mvg_pars = get_MLE_Gaussian_parameters(D,L)
    ret_pars = []

    Sw = get_withinCovariance(D, L)
    #identity = np.ones(D.shape[0])
    positions = [i for i in range(0,D.shape[0])]
    if naive:
        
        m = np.zeros((D.shape[0], D.shape[0]))
        diag = np.diag(Sw)
        m[positions, positions] = diag[positions]
        Sw=m
    for i in range(0,np.max(L)+1):
        mu = mvg_pars[i][0]
        cov = np.copy(Sw)
        ret_pars.append((mu,cov))

    return ret_pars

def get_MLE_NaiveGaussian_parameters(D: np.ndarray, L: np.ndarray ) -> List[Tuple[np.ndarray, np.ndarray]]:
    mvg_pars=get_MLE_Gaussian_parameters(D, L)
    ret_pars= []
    
    #identity= np.identity(D.shape[0])
    positions = [i for i in range(0,D.shape[0])]
    for i in range(0,np.max(L)+1):
        mu = mvg_pars[i][0]
        cov = np.zeros((D.shape[0], D.shape[0]))
        diag = np.diag(mvg_pars[i][1])
        cov[positions, positions] = diag[positions]
        ret_pars.append((mu,cov))
    
    return ret_pars
