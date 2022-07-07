from src_lib import *
from data import *

import sklearn

import scipy.stats as scz

FOLDS = 5
VERBOSE = True

class ExperimentsData:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))
        

    def scatter_features_raw(self):
        print(self.DTR.mean(1))
        print(self.DTR.std(1))
        myScatter(self.DTR[0:2, :], 2, self.LTR)
    
    def scatter_features_Z(self):
        preProc = Znorm()
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        print(DTR.mean(1))
        print(DTR.std(1))
        #print(DTR)
        if VERBOSE:
            myScatter(DTR[0:2, :], 2, LTR)

    def scatter_features_PCA_ZvsR(self):
        preProc1 = Znorm()
        preProc1.addNext(PCA(8))
        DTR1, LTR1 = preProc1.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            myScatter(DTR1[0:2, :], 2, LTR1)
        
        preProc2 = PCA(8)
        DTR2, LTR2 = preProc2.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            myScatter(DTR2[0:2, :], 2, LTR2)

    def plot_features_raw(self):
        myHistogram(self.DTR, 2, self.LTR)

    def plot_features_Gauss(self):
        preProc = Gaussianize()
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            myHistogram(DTR, 2, LTR)
    
    def plot_features_Z(self):
        preProc = Znorm()
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            myHistogram(DTR, 2, LTR)
    
    def plot_features_Z_PCA_8(self):
        preProc = Znorm()
        preProc.addNext(PCA(8))
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            myHistogram(DTR, 2, LTR)


    def plot_correlation_mat(self):
        mat = compute_Pearson_corr(self.DTR)
        plot_heatmap(mat)
    
    def plot_correlation_gauss(self):
        preproc = Gaussianize()
        DTR, LTR = preproc.learn(self.DTR, self.LTR)
        mat = compute_Pearson_corr(DTR)

        plot_heatmap(mat)

    def plot_correlation_Z(self):
        preproc = Znorm()
        DTR, LTR = preproc.learn(self.DTR, self.LTR)
        mat = compute_Pearson_corr(DTR)

        plot_heatmap(mat)
    
    def random_test(self):
        preproc = Znorm()
        DTR, LTR = preproc.learn(self.DTR, self.LTR)
        
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(self.DTR)
        A= scaler.transform(self.DTR)

        B = scz.zscore(self.DTR, axis = 1)

        print(DTR[:, 0:3])
        print("---")
        print(A[:, 0:3])
        print("---")
        print(B[:, 0:3])
        print("---")
        print((DTR-A)[:, 0:3])

        print("---")
        print (scaler.mean_.shape)
        print(DTR.mean())
        print(A.mean())
    
    def test_PCA_Z(self):
        preproc1 = Znorm()
        preproc2 = PCA(8)
        preprocSame = Znorm()
        preprocSame.addNext(PCA(8))

        DTRN, LTRN = preproc1.learn(self.DTR, self.LTR)
        DTRN, LTRN = preproc2.learn(DTRN, LTRN)

        DTRS, LTRS = preprocSame.learn(self.DTR, self.LTR)

        print((DTRN- DTRS)[:, 0:3])
    
    def plot_RawvsZ(self):
        preProc = Znorm()
        #preProc.addNext(PCA(8))
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            for i in range(0, DTR.shape[1]):
                myHistogram(vrow(DTR[i, :]), 2, LTR)
                myHistogram(vrow(self.DTR[i, :]), 2 , LTR)


if __name__ == "__main__":
    exps = ExperimentsData("gend")

    #exps.plot_features_Z_PCA_8()
    #exps.random_test()
    #exps.test_PCA_Z()

    #exps.scatter_features_raw()
    #exps.scatter_features_Z()
    exps.scatter_features_PCA_ZvsR()
    #exps.plot_features_raw()
    #exps.plot_RawvsZ()
    
