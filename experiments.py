from src_lib import *
from data import *


class Experiments:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)

    def plot_features_raw(self):
        myHistogram(self.DTR, 2, self.LTR)

    def plot_features_Gauss(self):
        preProc = Gaussianize()
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        #print(DTR)
        myHistogram(DTR, 2, LTR)
    
    def plot_correlation_mat(self):
        mat = compute_Pearson_corr(self.DTR)
        plot_heatmap(mat)



if __name__ == "__main__":
    exps = Experiments("gend")

    #exps.plot_features_raw()
    #exps.plot_features_Gauss()
    exps.plot_correlation_mat()
        