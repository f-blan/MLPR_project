import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools as itt
from typing import List

def myHistogram(data: np.ndarray,n_labels:int,labels:np.ndarray, bins: int=20):
    colors = ["red", "green", "blue","orange" , "gray", "orange"]

    for i in range(0,data.shape[0]):
        for j in range(0,n_labels):
            plt.hist(data[i, labels == j],bins=bins, density=True, color=colors[j],alpha=0.6, histtype='bar', rwidth=0.9)
        plt.show()

def plot_vals(dataY: List[List[float]], dataX: List[float], logplot: bool = True, compare_mode: bool = False):
    colors = ["red", "green", "blue","orange" , "gray", "yellow"]

    plot_fun = plt.semilogx if logplot else plt.plot

    for i, app in enumerate(dataY):
        if compare_mode == False:
            plot_fun(dataX, app, color = colors[i])
        else:
            if i%2 == 0:
                plot_fun(dataX, app, color = colors[i])
            else:
                plot_fun(dataX, app, color=colors[i-1], linestyle='dashed')

    
    plt.show()

def plot_heatmap(D: np.ndarray):
    x = [i for i in range(0, D.shape[0])]
    y = [i for i in range(0, D.shape[0])]

    fig, ax = plt.subplots()
    im = ax.imshow(D)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x)), labels=x)
    ax.set_yticks(np.arange(len(y)), labels=y)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x)):
        for j in range(len(y)):
            text = ax.text(j, i, round(D[i, j], 2),
                           ha="center", va="center", color="w")

    ax.set_title("Heatmap")
    fig.tight_layout()
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

