import numpy as np




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