from src_lib import *
from data import *
from src_lib.GMM_Model import GMMLBG_DT_Model, GMMLBG_Diag_Model, GMMLBG_Model, GMMLBG_Tied_Model
from src_lib.SVM_Model import Kernel

FOLDS = 5
VERBOSE = True
STOP_TH = 1e-3

class ExperimentsGMM:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))
    
    def find_best_K_FC_raw(self):
        
        minDCFList = []
        accuracies = []
        model = GMMLBG_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)
        #0.0440 - 0.1196 - 0.1166
    
    def find_best_K_FC_PCA8(self):
        
        minDCFList = []
        accuracies = []
        preproc = PCA(8)
        
        model = GMMLBG_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        print(f"model pars: {len(model.pars) - {len(model.pars[0])}}")
        
        model = GMMLBG_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        print(f"model pars: {len(model.pars)} - {len(model.pars[0])}")

        model = GMMLBG_Model(2,STOP_TH, 1, preProcess= preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)

    def plot_K_FC_PCA8(self):
        minDCFList = [[0.0443, 0.0443, 0.0376, 0.0386, 0.0523], [0.1226, 0.1193, 0.1066, 0.11366, 0.1336], [0.1399, 0.1236, 0.1053, 0.1063, 0.1293]]
        vals = [ 2, 4, 8, 16, 32]
        plot_vals(minDCFList, vals)

    def find_best_K_FC_Gauss(self):
        
        minDCFList = []
        accuracies = []
        preproc = Gaussianize()
        
        model = GMMLBG_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        print(f"max exps: {model.n_gauss_exp}")

        model = GMMLBG_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        model = GMMLBG_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)

    def plot_K_FC_Gauss(self):
        minDCFList = [[0.0530, 0.0456, 0.0486, 0.0560, 0.0656], [0.1373, 0.1246, 0.1246, 0.11416, 0.1956], [0.1703, 0.1283, 0.1286, 0.1450, 0.1930]]
        vals = [ 2, 4, 8, 16, 32]
        plot_vals(minDCFList, vals)
        

    def find_best_K_T_raw(self):
        
        minDCFList = []
        accuracies = []
        model = GMMLBG_Tied_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)
            #0.0320 - 0.0916 - 0.1063

    def find_best_K_T_PCA8(self):
        
        minDCFList = []
        accuracies = []
        preproc = PCA(8)
        
        model = GMMLBG_Tied_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        model = GMMLBG_Tied_Model(2,STOP_TH, 1, preProcess = preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    def plot_K_T_PCA8(self):
        minDCFList = [[0.0670, 0.0670, 0.0849, 0.0996, 0.0803],[0.1600, 0.1590, 0.0206, 0.2956,0.2066],[0.1413, 0.1353,0.1156,0.1100,0.1153]]

    
    def find_best_K_T_Gauss(self):
        
        minDCFList = []
        accuracies = []
        preproc = Gaussianize()
        
        model = GMMLBG_Tied_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Tied_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)
            #0.0579 - 0.1406 - 0.1669
    
    def find_best_K_N_raw(self):
        
        minDCFList = []
        accuracies = []
        
        model = GMMLBG_Diag_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Diag_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Diag_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)
            #0.0819 - 0.2300 - 0.2289

    def find_best_K_N_PCA8(self):
        
        minDCFList = []
        accuracies = []
        preproc = PCA(8)
        
        model = GMMLBG_Diag_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Diag_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Diag_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    def plot_K_N_raw(self):
        minDCFList = [[0.0666, 0.7166, 0.0783, 0.0863, 0.0926], [0.156, 0.1716, 0.2030, 0.2640, 0.2470]]
    
    def find_best_K_N_Gauss(self):
        
        minDCFList = []
        accuracies = []
        preproc = Gaussianize()
        
        model = GMMLBG_Diag_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Diag_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_Diag_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    def find_best_K_NT_raw(self):
        
        minDCFList = []
        accuracies = []
        
        model = GMMLBG_DT_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_DT_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_DT_Model(2,STOP_TH, 1)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)
            #0.0819 - 0.2300 - 0.2289

    def find_best_K_NT_PCA8(self):
        
        minDCFList = []
        accuracies = []
        preproc = PCA(8)
        
        model = GMMLBG_DT_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_DT_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_DT_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    
    def find_best_K_NT_Gauss(self):
        
        minDCFList = []
        accuracies = []
        preproc = Gaussianize()
        
        model = GMMLBG_DT_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.bal_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_DT_Model(2,STOP_TH, 1,preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.female_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = GMMLBG_DT_Model(2,STOP_TH, 1, preProcess=preproc)#GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(1, 5), logbase=2, n_vals = 5, logBounds=False, e_prior=self.male_app[0])
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    
    

if __name__ == "__main__":
    exps = ExperimentsGMM("gend")

    exps.find_best_K_FC_raw()
    #exps.find_best_K_FC_Gauss()
    #exps.find_best_K_FC_PCA8()
    #exps.plot_K_FC_PCA8()
    #exps.find_best_K_T_raw()
    #exps.find_best_K_T_PCA8()
    #exps.find_best_K_T_Gauss()
    #exps.find_best_K_N_raw()
    #exps.find_best_K_N_PCA8()
    #exps.find_best_K_N_Gauss()
    #exps.find_best_K_NT_raw()
    #exps.find_best_K_NT_PCA8()
    #exps.find_best_K_NT_Gauss()