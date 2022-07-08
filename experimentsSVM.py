from src_lib import *
from data import *
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
    
    def find_best_C_L_raw(self):
        minDCFList = []
        accuracies = []
        model = SVML_Model(2, 1, 0.1)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        model = SVML_Model(2, 1, 0.1)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        
        model = SVML_Model(2, 1, 0.1)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

        #best: 0.0556 - 0.1516 - 0.1486

    def plot_C_L(self):
        #for some reason plot didn't work
        minDCFList = [[0.0963, 0.1473, 0.0556, 0.0770, 0.7026], [0.2216, 0.5183, 0.1516, 0.1750, 0.8090], [0.254, 0.4710, 0.1486, 0.2403, 0.8896,]]
        vals = [1e-03, 3.1622e-02, 1.0, 3.1622e+01, 1e+03]

        plot_vals(minDCFList, vals)

    def find_best_C_L_Gauss(self):
        minDCFList = []
        accuracies = []
        preproc = Gaussianize()
        model = SVML_Model(2, 1, 0.1, preProcess= preproc)

        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)
        
        model = SVML_Model(2, 1, 0.1, preProcess= preproc)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        
        model = SVML_Model(2, 1, 0.1, preProcess= preproc)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)
            # 0.0550 - 0.1586 - 0.1566
    


    def find_best_C_L_PCA8(self):
            minDCFList = []
            accuracies = []
            preproc = PCA(8)
            model = SVML_Model(2, 1, 0.1, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)
        
            model = SVML_Model(2, 1, 0.1, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            model = SVML_Model(2, 1, 0.1, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)
            #best: 0.0683 - 0.1886 - 0.1786

    def find_best_C_Quad_raw(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="poly2", c = 1)
            
            model = SVMNL_Model(2, 0, 0.1, kernel=kernel)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)
        
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

       
            model = SVML_Model(2, 1, 0.1, kernel= kernel)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                # array = [0.5376, 0.5783, 0.6703, 0.6546, 0.7256]
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)

    def find_best_C_Quad_Gauss(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="poly2", c = 1)
            
            preproc = Gaussianize()
            model = SVMNL_Model(2, 0, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)
        
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

       
            model = SVMNL_Model(2, 1, 0.1, kernel= kernel, preProcess= preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)
                # 0.0546 - 0.1506 - 0.1670 

    def find_best_C_Quad_PCA8(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="poly2")
            preproc = PCA(8)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)
        
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

       
            model = SVML_Model(2, 1, 0.1, kernel= kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)

    def find_best_C_RBF_raw_bal(self):
        minDCFList = []
        accuracies = []
        kernel = Kernel(kname="RBF", gamma = 0.001)
            
        model = SVMNL_Model(2, 1, 0.1, kernel=kernel)    
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

            
        kernel = Kernel(kname="RBF", gamma = 0.01)
        model = SVMNL_Model(2, 1, 0.1, kernel=kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

            
        kernel = Kernel(kname="RBF", gamma = 0.1)
        model = SVMNL_Model(2, 1, 0.1, kernel= kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)
        
        #best: 0.0520 - 0.0513- 0.4286
    def find_best_C_RBF_PCA8_bal(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="RBF", gamma = 0.001)
            preproc = PCA(8)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            
            kernel = Kernel(kname="RBF", gamma = 0.01)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)


            
            kernel = Kernel(kname="RBF", gamma = 0.1)
            model = SVMNL_Model(2, 1, 0.1, kernel= kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)
    
    def plot_RBF_pca8_bal(self):
        #for some reason plot didn't work
        minDCFList = [[0.0543, 0.0543, 0.0543, 0.0543, 0.4519], [0.0523, 0.0516, 0.0516, 0.0533, 0.2943], [0.1846, 0.6146, 0.7973, 0.6963, 0.6386,]]
        vals = [1e-03, 3.1622e-02, 1.0, 3.1622e+01, 1e+03]

        plot_vals(minDCFList, vals)
    
    def find_best_C_RBF_Gauss_bal(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="RBF", gamma = 0.001)
            preproc = Gaussianize()
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            
            kernel = Kernel(kname="RBF", gamma = 0.01)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)


            
            kernel = Kernel(kname="RBF", gamma = 0.1)
            model = SVMNL_Model(2, 1, 0.1, kernel= kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.bal_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)
                #0.2870 - 0.0996 - 0.0673
    

    def find_best_C_RBF_raw_female(self):
        minDCFList = []
        accuracies = []
        kernel = Kernel(kname="RBF", gamma = 0.001)
            
        model = SVMNL_Model(2, 1, 0.1, kernel=kernel)    
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

            
        kernel = Kernel(kname="RBF", gamma = 0.01)
        model = SVMNL_Model(2, 1, 0.1, kernel=kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

            
        kernel = Kernel(kname="RBF", gamma = 0.1)
        model = SVMNL_Model(2, 1, 0.1, kernel= kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    def find_best_C_RBF_PCA8_female(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="RBF", gamma = 0.001)
            preproc = PCA(8)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            
            kernel = Kernel(kname="RBF", gamma = 0.01)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)


            
            kernel = Kernel(kname="RBF", gamma = 0.1)
            model = SVMNL_Model(2, 1, 0.1, kernel= kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.female_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)

    def find_best_C_RBF_raw_male(self):
        minDCFList = []
        accuracies = []
        kernel = Kernel(kname="RBF", gamma = 0.001)
            
        model = SVMNL_Model(2, 1, 0.1, kernel=kernel)    
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

            
        kernel = Kernel(kname="RBF", gamma = 0.01)
        model = SVMNL_Model(2, 1, 0.1, kernel=kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

            
        kernel = Kernel(kname="RBF", gamma = 0.1)
        model = SVML_Model(2, 1, 0.1, kernel= kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
        minDCFList.append(minDCFs)
        accuracies.append(accs)

        if VERBOSE:
            plot_vals(minDCFList, vals)
            plot_vals(accuracies, vals)

    def find_best_C_RBF_PCA8_male(self):
            minDCFList = []
            accuracies = []
            kernel = Kernel(kname="RBF", gamma = 0.001)
            preproc = PCA(8)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            
            kernel = Kernel(kname="RBF", gamma = 0.01)
            model = SVMNL_Model(2, 1, 0.1, kernel=kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)


            
            kernel = Kernel(kname="RBF", gamma = 0.1)
            model = SVMNL_Model(2, 1, 0.1, kernel= kernel, preProcess=preproc)
            kcv = KCV(model, 5)
            minDCFs, vals, accs = kcv.find_best_par(model, self.DTR, self.LTR, 0,(-3, 3), e_prior=self.male_app[0], n_vals=5)
            minDCFList.append(minDCFs)
            accuracies.append(accs)

            if VERBOSE:
                plot_vals(minDCFList, vals)
                plot_vals(accuracies, vals)

if __name__ == "__main__":
    exps = ExperimentsSVM("gend")

    #exps.find_best_C_L()
    #exps.plot_C_L()
    #exps.find_best_C_L_PCA8()
    #exps.find_best_C_L_Gauss()
    #exps.find_best_C_Quad_PCA8()
    #exps.find_best_C_RBF_PCA8_bal()
    #exps.plot_RBF_pca8_bal()
    #exps.find_best_C_Quad_raw()
    #exps.find_best_C_Quad_Gauss()
    #exps.find_best_C_RBF_raw_bal()
    #exps.find_best_C_RBF_raw_female()
    #exps.find_best_C_RBF_PCA8_female()
    #exps.find_best_C_RBF_raw_male()
    #exps.find_best_C_RBF_PCA8_male()
    exps.find_best_C_RBF_Gauss_bal()