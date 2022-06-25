from src_lib import *
from data import *

FOLDS = 5
VERBOSE = True

class Experiments:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))

    def plot_features_raw(self):
        myHistogram(self.DTR, 2, self.LTR)

    def plot_features_Gauss(self):
        preProc = Gaussianize()
        DTR, LTR = preProc.learn(self.DTR, self.LTR)
        #print(DTR)
        if VERBOSE:
            myHistogram(DTR, 2, LTR)
    
    def plot_correlation_mat(self):
        mat = compute_Pearson_corr(self.DTR)
        plot_heatmap(mat)
    
    def MVG_FC_Raw(self):
        model = MVG_Model(2, False, False)
        kcvw = KCV(model, FOLDS)
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_RAW: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_RAW: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_RAW: {minDCFm}, th: {_m}")
            #last res: 0.049 - 0.1206 - 0.1266
    
    def MVG_FC_GAUSS(self):
        preproc = Gaussianize()
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_RAW: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_RAW: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_RAW: {minDCFm}, th: {_m}")
            #last res: 0.627 - 0.953 - 0.998 # may be much worse because gaussianization destroys correlation between features







if __name__ == "__main__":
    exps = Experiments("gend")

    #exps.plot_features_raw()
    #exps.plot_features_Gauss()
    #exps.plot_correlation_mat()

    #exps.MVG_FC_Raw()
    #exps.MVG_FC_GAUSS()
        