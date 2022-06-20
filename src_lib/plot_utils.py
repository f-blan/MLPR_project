import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools as itt
from typing import List

def myHistogram(data: np.ndarray,n_labels:int,labels:np.ndarray, bins: int=20):
    colors = ["red", "green", "yellow", "blue", "gray", "orange"]

    for i in range(0,data.shape[0]):
        for j in range(0,n_labels):
            plt.hist(data[i, labels == j],bins=bins, density=True, color=colors[j],alpha=0.6)
        plt.show()

def plot_vals(dataY: List[List[float]], dataX: List[float], logplot: bool = True):
    colors = ["red", "green", "yellow", "blue", "gray", "orange"]

    for i, app in enumerate(dataY):
        plt.semilogx( dataX, app, color= colors[i])
    
    plt.show()
        

def myScatter(data: np.ndarray, n_labels:int,labels:np.ndarray):
    colors = ["red", "green", "yellow", "blue", "gray", "orange"]
    n_feats = data.shape[0]
    pairs = itt.combinations([i for i in range(0, n_feats)],2)
    
    for p in pairs:
        for j in range(0,n_labels):
            plt.scatter(data[p[0], labels == j], data[p[1], labels==j], color=colors[j], alpha=0.6)
        plt.show()

def myScatter1d(data:np.ndarray, n_labels:int, labels:np.ndarray):
    colors = ["red", "green", "yellow", "blue", "gray", "orange"]
    n_feats = 1

    for j in range(0,n_labels):
            plt.scatter(data[0, labels == j], data[0, labels==j], color=colors[j],alpha=0.6)
    plt.show()

