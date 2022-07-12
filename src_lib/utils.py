from cmath import sqrt
import numpy as np
import random as r

def compute_Pearson_corr(D: np.ndarray):
    cov = get_covarianceCentered(D)

    sqrt_vars =np.sqrt( np.var(D, axis = 1) )

    #not very relevant function, implement with loop
    for y in range(0,cov.shape[0]):
        for x in range(0,cov.shape[1]):
            cov[y,x] = cov[y,x]/(sqrt_vars[x]*sqrt_vars[y])
    
    return cov


def compute_acc(predicted: np.ndarray, real: np.ndarray) -> float:
    acc = predicted == real

    return np.sum(acc)/acc.shape[0], np.sum(acc)


def get_covariance(D: np.ndarray) -> np.ndarray:
    return np.dot(D, D.T)/(D.shape[1])

def get_covarianceCentered(D: np.ndarray) -> np.ndarray:
    mu = D.mean(1)
    DC = D - mu.reshape((mu.shape[0], 1))
    return np.dot(DC, DC.T)/(DC.shape[1])

def get_betweenCovariance(D: np.ndarray, labels: np.ndarray) -> np.ndarray:
    SB = np.zeros((D.shape[0], D.shape[0]))
    mu = np.mean(D, axis=1)
    

    for i in range(0, np.max(labels)+1):
        nc = D[:, labels==i].shape[1]
        mu_c = (np.mean(D[:, labels == i], axis=1) - mu)
        mu_c = mu_c.reshape((mu_c.shape[0], 1))
        to_add = nc*np.dot(mu_c, mu_c.T)
        
        SB += to_add
    return SB/D.shape[1]


def get_withinCovariance(D: np.ndarray, labels: np.ndarray) -> np.ndarray:
    SW = np.zeros((D.shape[0], D.shape[0]))
    
    for i in range(0, np.max(labels)+1):
        Dc = D[:, labels == i]
        SW += Dc.shape[1] * get_covarianceCentered(Dc)
    
    return SW/D.shape[1]

def joint_diagonalization(SB: np.ndarray, SW: np.ndarray, m: int) ->np.ndarray:
    U,s,_ = np.linalg.svd(SW)

    P1 = np.dot(U*vrow(1.0/(s**0.5)), U.T)

    SBT = np.dot(P1, SB)
    SBT = np.dot(SBT, P1.T)

    s2, U2 = np.linalg.eigh(SBT)

    P2 = U2[:, ::-1][:, 0:m]

    return np.dot(P1.T, P2)

def loglikelihood(x:np.ndarray, mu:np.ndarray, C: np.ndarray, logdistribution= lambda x,mu, C: logpdf_GAU_ND(x,mu,C)) -> float:

    return logdistribution(x,mu,C).sum()

def logpdf_GAU_ND_Opt(X: np.ndarray, mu: np.ndarray, C:np.ndarray) -> np.ndarray:
    P = np.linalg.inv(C)
    const = -0.5*X.shape[0] *np.log(2*np.pi)
    const += -0.5*np.linalg.slogdet(C)[1]

    Y=[]

    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5*np.dot((x-mu).T, np.dot(P, (x-mu)))
        Y.append(res)
    return np.array(Y).ravel()

def vcol(vec: np.ndarray) -> np.ndarray:
    return vec.reshape((vec.size,1))
def vrow(vec: np.ndarray) -> np.ndarray:
    return vec.reshape((1,vec.size))

def shuffle_and_split_dataset(D: np.ndarray, L: np.ndarray, dims = 1):
    if dims == 1:
        shuffled_zip = list(([(D[i], L[i]) for i in range(0,L.shape[0])]))
    else:
        shuffled_zip = list(([(D[:, i], L[i]) for i in range(0,L.shape[0])]))
    length = len(shuffled_zip)
    r.Random(5).shuffle(shuffled_zip)

    half_length = int(length*0.8)

    t_split = shuffled_zip[0: half_length]
    v_split = shuffled_zip[ half_length :]

        
    t_S = np.array( list(map(lambda x: x[0],  t_split)))
    t_L = np.array(list(map(lambda x: x[1], t_split)))
    v_S = np.array(list(map(lambda x: x[0],  v_split)))
    v_L = np.array(list(map(lambda x: x[1], v_split)))

    return (t_S.T, t_L), (v_S.T, v_L) 