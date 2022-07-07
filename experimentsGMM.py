from src_lib import *
from data import *
from src_lib.GMM_Model import GMMLBG_Model, GMMLBG_Tied_Model
from src_lib.SVM_Model import Kernel

FOLDS = 5
VERBOSE = True

class ExperimentsSVM:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))
    
    def find_best_K_FC_raw(self):
        
        minDCFList = []
        accuracies = []
        model = GMMLBG_Model(2,1e-2, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Model(2,1e-2, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Model(2,1e-2, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)
    
    def find_best_K_FC_PCA8(self):
        
        minDCFList = []
        accuracies = []
        preproc = PCA(8)
        model = GMMLBG_Model(2,1e-2, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Model(2,1e-2, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Model(2,1e-2, 1, preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)
    
    def find_best_K_T_raw(self):
        
        minDCFList = []
        accuracies = []
        model = GMMLBG_Tied_Model(2,1e-2, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,1e-2, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,1e-2, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)
    
    def find_best_K_T_PCA8(self):
        
        minDCFList = []
        accuracies = []
        preproc = PCA(8)
        
        model = GMMLBG_Tied_Model(2,1e-2, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,1e-2, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,1e-2, 1, preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)
