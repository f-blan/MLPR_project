from src_lib import *
from data import *

FOLDS = 5
VERBOSE = True

class ExperimentsLR:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))
    
    def find_best_lambda(self):
        minDCFList = []
        accuracies = []
        model = LRBinary_Model(2, 0.1, rebalance=False)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=10)
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        model = LRBinary_Model(2, 0.1, rebalance=False)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=10)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = LRBinary_Model(2, 0.1, rebalance=False)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=10)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)

        #last res: best is low 0.4666 - 0.1233 - 0.1253
    
    def find_best_lambda_PCA_8(self):
        minDCFList = []
        accuracies = []
        preProc = PCA(8)
        model = LRBinary_Model(2, 0.1, rebalance=False, preProcess=preProc)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=10)
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        model = LRBinary_Model(2, 0.1, rebalance=False, preProcess= preProc)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=10)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        model = LRBinary_Model(2, 0.1, rebalance=False, preProcess=preProc)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=10)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        plot_vals(minDCFList, vals)
        plot_vals(accuracies, vals)


    
if __name__ == "__main__":
    exps = ExperimentsLR("gend")

    #exps.find_best_lambda()
    exps.find_best_lambda_PCA_8()
