import numpy as np
import scipy as sp
from typing import Tuple, List

from src_lib import *


class GMMLBG_Model(Model):
    def __init__(self, n_classes: int, stop_threshold: float, n_gauss_exp: int, alpha: float = 0.1, preProcess: PreProcess = PreProcess("None"), constrained: bool = True,verbose: bool = False, bound: float = 0.01):
        super().__init__(n_classes, preProcess=preProcess)
        self.stop_threshold = stop_threshold
        self.n_gauss_exp = n_gauss_exp
        self.alpha = alpha
        self.verbose = verbose
        self.constrained = constrained
        self.bound = bound
        self.verbose = verbose

    def _compute_responsibility(self, X:np.ndarray, gmm: List[Tuple[float, np.ndarray, np.ndarray]]) -> np.ndarray:
        Sj = np.zeros((len(gmm), X.shape[1]))
        i=0
        for  w, mu, C in gmm:
            Sj[i, :]= logpdf_GAU_ND_Opt(X, mu, C) + np.log(w)
            i+=1
        #print(ret.shape)
        Sm = sp.special.logsumexp(Sj, axis=0)
        ll = Sm.sum()/X.shape[1]
        ret = np.exp(Sj- Sm)
        return ret, ll  

    def _get_next_params(self, responsibilities: np.ndarray, X: np.ndarray) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        nextGMM = []

        for g in range(0, responsibilities.shape[0]):
            resp = responsibilities[g, :]
            Z= resp.sum()
            F= (vrow(resp)*X).sum(1)
            S= np.dot(X, (vrow(resp)*X).T)
            w = Z/X.shape[1]
            mu = vcol(F/Z)
            C = S/Z - np.dot(mu, mu.T)
            if self.constrained:
                U, s, _ = np.linalg.svd(C)
                s[s<self.bound] = self.bound
                C = np.dot(U, vcol(s)*U.T)

            nextGMM.append((w,mu,C))
        
        return nextGMM

    def _EMtrain(self, X: np.ndarray, L: np.ndarray, gmm_init: list)-> List[Tuple[float, np.ndarray, np.ndarray]]:
        curr_ll = None
        prev_ll = None 
        gmm = gmm_init

        while prev_ll is None or curr_ll-prev_ll > self.stop_threshold:
            prev_ll = curr_ll

            resp, curr_ll = self._compute_responsibility(X, gmm)
            gmm = self._get_next_params(resp, X)
            
            if self.verbose: print(curr_ll)
            
        return gmm
    
    def _train(self, X:np.ndarray, L:np.ndarray) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        
        C = get_covariance(X)
        
        if self.constrained:
            U, s, _ = np.linalg.svd(C)
            s[s<self.bound] = self.bound
            C = np.dot(U, vcol(s)*U.T)

        gmm = [(1.0,vcol(X.mean(1)), C)]
        
        for i in range(0, self.n_gauss_exp):
            #gmm_opt = self._EMtrain(X, L, gmm)
            gmm_2g = []
            for w, mu, C in gmm:
                U,s,Vh = np.linalg.svd(C)
                d = U[:, 0:1]*s[0]**0.5 * self.alpha
                gmm_2g.append((w*0.5, mu+d, C))
                gmm_2g.append((w*0.5, mu-d, C))
            
            gmm = self._EMtrain(X,L,gmm_2g)
        return gmm
    
    def train(self, X:np.ndarray, L:np.ndarray):
        X, L = self.preProcess.learn(X, L)
        self.pars: List[List[Tuple[float, np.ndarray, np.ndarray]]] = []

        for i in range(0, self.n_classes):
            D = X[:, L == i]
            #print(D)
            self.pars.append(self._train(D, np.zeros(1)))
            
        
        if self.verbose: print(f"training is over {len(self.pars)}")
        
    def _logpdf_GMM(self, X:np.ndarray, gmm: list) -> float:
        Sj = np.zeros((len(gmm), X.shape[1]))
        i=0
        for  w, mu, C in gmm:
            Sj[i, :]= logpdf_GAU_ND_Opt(X, mu, C) + np.log(w)
            i+=1
        #print(ret.shape)
        Sm = sp.special.logsumexp(Sj, axis=0)
        ll = Sj.sum(axis = 0)/X.shape[1]
        #print(ll.shape)
        ret = np.exp(Sj- Sm)
        return Sm, ll  
    
    
    def _get_score_matrix(self, DTE: np.ndarray, pars, log_scores = False) -> np.ndarray:
        n_classes = len(pars)
        vecs=[]

        for i in range(0,self.n_classes):
        
            
            vec, _  =self._logpdf_GMM(DTE, pars[i])
            #print(f"vec shape: {vec.shape}")
            vecs.append(vec) if log_scores else vecs.append(np.exp(vec))

    
        return np.array(vecs)
    
    def predict(self, D: np.ndarray, L: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        D, L = self.preProcess.apply(D,L)
        logS= self._get_score_matrix(D, self.pars, log_scores=False)
        #print(logS.shape)
        logSjoint= logS+ np.log(self.prior)
        logSMarginal= vrow(sp.special.logsumexp(logSjoint))
        logSPost = logSjoint - logSMarginal
        logPredL = np.argmax(logSPost, axis=0)
        acc, corrects = compute_acc(logPredL, L)

        #print(logSPost.shape)
        return acc, logPredL, logS[1, :]/logS[0, :]


class GMMLBG_Diag_Model(GMMLBG_Model):
    def __init__(self, n_classes: int, stop_threshold: float, n_gauss_exp: int, alpha: float = 0.1, preProcess: PreProcess = PreProcess("None"), constrained: bool = True, verbose: bool = False, bound: float = 0.01):
        super().__init__(n_classes, stop_threshold, n_gauss_exp, alpha, preProcess, constrained, verbose, bound=bound)
    
    def _get_next_params(self, responsibilities: np.ndarray, X: np.ndarray) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        nextGMM = []

        for g in range(0, responsibilities.shape[0]):
            resp = responsibilities[g, :]
            Z= resp.sum()
            F= (vrow(resp)*X).sum(1)
            S= np.dot(X, (vrow(resp)*X).T)
            w = Z/X.shape[1]
            mu = vcol(F/Z)
            C = S/Z - np.dot(mu, mu.T)

            dim = C.shape[0]
            m = np.zeros((dim, dim))
            diag = np.diag(C)
            positions = [i for i in range(0,dim)]
            m[positions, positions] = diag[positions]
            C=m
            if self.constrained:
                U, s, _ = np.linalg.svd(C)
                s[s<self.bound] = self.bound
                C = np.dot(U, vcol(s)*U.T)

            nextGMM.append((w,mu,C))
        
        return nextGMM


class GMMLBG_Tied_Model(GMMLBG_Model):
    def __init__(self, n_classes: int, stop_threshold: float, n_gauss_exp: int, alpha: float = 0.1, preProcess: PreProcess = PreProcess("None"), constrained: bool = True, verbose: bool = False, bound: float = 0.01):
        super().__init__(n_classes, stop_threshold, n_gauss_exp, alpha, preProcess, constrained, verbose, bound=bound)
    
    def _get_next_params(self, responsibilities: np.ndarray, X: np.ndarray) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        nextGMM : List[Tuple[float, np.ndarray, np.ndarray]] = []
        computed = False
        
        for g in range(0, responsibilities.shape[0]):
            resp = responsibilities[g, :]
            Z= resp.sum()
            F= (vrow(resp)*X).sum(1)
            S= np.dot(X, (vrow(resp)*X).T)
            w = Z/X.shape[1]
            mu = vcol(F/Z)
            
            if computed == False:
                tied_C = S/Z - np.dot(mu, mu.T)
                computed = True
            C = tied_C.copy()
            
                
            dim = C.shape[0]
            m = np.zeros((dim, dim))
            diag = np.diag(C)
            positions = [i for i in range(0,dim)]
            m[positions, positions] = diag[positions]
            C=m
            

            nextGMM.append((w,mu,C))
        

    
        dim = nextGMM[0][2].shape[0]
        sumC =np.zeros((dim, dim))
        for i in range(0,len(nextGMM)):
            sumC += nextGMM[i][1] *nextGMM[i][2]

        if self.constrained:
            sumC = sumC/len(nextGMM)
            U, s, _ = np.linalg.svd(sumC)
            s[s<self.bound] = self.bound
            sumC = np.dot(U, vcol(s)*U.T)

        for i in range(0,len(nextGMM)):
                
            nextGMM[i] = (nextGMM[i][0],nextGMM[i][1], sumC )
        
        return nextGMM
   
    
    