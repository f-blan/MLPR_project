import numpy as np


import sklearn.datasets

import json

import random as r
def load_data_comm():

    lInf = []

    f=open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f=open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f=open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()
    
    return lInf, lPur, lPar

def split_data_comm(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, lTest

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D: np.ndarray, L: np.ndarray, seed: int=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    #print(L)
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def load_ds(filename: str, separator: str = ",", shuffle = False, seed = 1):
    f=open(f"data/{filename}")

    vecs = []
    labels = []
    lines = []
    for line in f:
        lines.append(line)

    if shuffle:
        r.Random(seed).shuffle(lines)
        

    for line in lines:
        words = line.split(separator)
        l= words[-1].strip()

        feats= [float(i) for i in words[0:-1]]
        vecs.append(np.array(feats))
        labels.append(int(l))

    f.close()
    return np.transpose(np.array(vecs)), np.array(labels)

def load_Gender(shuffle = False):
    DTR, LTR = load_ds("GenderTrain.txt",", ", shuffle=shuffle)
    DTE, LTE = load_ds("GenderTest.txt", ", ", shuffle=shuffle)

    return (DTR, LTR), (DTE, LTE)