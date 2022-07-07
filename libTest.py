from src_lib import *
from data import *
import unittest
from src_lib.GMM_Model import *
from src_lib.SVM_Model import Kernel, SVML_Model, SVMNL_Model
from src_lib.BD_wrapper import *

from data.data_loader import * 

class LibTest():
    def __init__(self) -> None:
        
        self.D, self.L = load_iris()
        self.Db, self.Lb = load_iris_binary()

        (self.DTR, self.LTR), (self.DTE, self.LTE) = split_db_2to1(self.D, self.L)
        (self.DTRbin, self.LTRbin), (self.DTEbin, self.LTEbin) = split_db_2to1(self.Db, self.Lb)
        
        

    def test_PCA(self):

        pca = PCA(2)

        pca.learn(self.D, self.L)

        MD, L = pca.apply(self.D, self.L)
        myScatter(MD, 4,self.L)
    
    def test_PCA_bin(self):
        pca = PCA(2)

        pca.learn(self.Db, self.Lb)

        MD, L = pca.apply(self.Db, self.Lb)
        myScatter(MD, 4,self.Lb)

    def test_LDA(self):
        lda = LDA(1)
        D,L=lda.learn(self.D, self.L)
        myScatter1d(D, 3, L)
    
    def test_LDA_bin(self):
        
        lda = LDA(1)
        D,L=lda.learn(self.Db, self.Lb)
        
        myScatter1d(D, 2, L)
    
    def test_PCA_LDA_Combined(self):
        root = PCA(3)

        root.addNext(LDA(1))

        D, L = root.learn(self.D, self.L)
        print(D.shape)
        myScatter1d(D,3,L)
    
    def test_MVG_Model_FC(self):
        model = MVG_Model(3, False, False )

        model.train(self.DTR, self.LTR)
        acc,_preds, scores = model.predict(self.DTE, self.LTE)

        print(acc)
        
    def test_MVG_Model_N(self):
        model = MVG_Model(3,False, True)

        model.train(self.DTR, self.LTR)
        acc,_preds, scores = model.predict(self.DTE, self.LTE)

        print(acc)

    def test_MVG_Model_T(self):
        model = MVG_Model(3, True, False )

        model.train(self.DTR, self.LTR)
        acc,_preds, scores = model.predict(self.DTE, self.LTE)

        print(acc)
    
    def test_MVG_Model_NT(self):
        model = MVG_Model(3, True, True )

        model.train(self.D, self.L)
        acc,_preds, scores = model.predict(self.DTE, self.LTE)

        print(acc)
    
    def test_KCV_FC(self):
        model = MVG_Model(3, False, False)
        kcv = KCV(model, -1, LOO = True)
        acc, _=kcv.crossValidate(self.D, self.L)
        print(1-acc)

    def test_KCV_N(self):
        model = MVG_Model(3, False, True)
        kcv = KCV(model, -1, LOO = True)
        acc,_=kcv.crossValidate(self.D, self.L)
        print(1-acc)
    
    def test_KCV_T(self):
        model = MVG_Model(3, True, False)
        kcv = KCV(model, -1, LOO = True)
        acc,_=kcv.crossValidate(self.D, self.L)
        print(1-acc)

    def test_KCV_NT(self):
        model = MVG_Model(3, True, True)
        kcv = KCV(model, -1, LOO = True)
        acc,_=kcv.crossValidate(self.D, self.L)
        print(1-acc)
    
    def test_LogReg(self):
        l = 10e-6
        model = LRBinary_Model(2, l)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)
        
        l = 10e-3
        model = LRBinary_Model(2, l)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)

        l = 0.1
        model = LRBinary_Model(2, l)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)

        l = 1
        model = LRBinary_Model(2, l)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)
    
    def test_LogReg_Rebalanced(self):
        l = 10e-6
        model = LRBinary_Model(2, l, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)
        
        l = 10e-3
        model = LRBinary_Model(2, l,rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)

        l = 0.1
        model = LRBinary_Model(2, l,rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)

        l = 1
        model = LRBinary_Model(2, l,rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)
    
    def test_SVMNL_RBF(self):
        kernel = Kernel("RBF",gamma=1.0)
        model = SVMNL_Model(2,0.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("RBF",gamma=10.0)
        model = SVMNL_Model(2,0.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("RBF",gamma=1.0)
        model = SVMNL_Model(2,1.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("RBF",gamma=10.0)
        model = SVMNL_Model(2,1.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)
    
    def test_SVMNL_Poly(self):
        kernel = Kernel("poly2",d=2, c=0)
        model = SVMNL_Model(2,0.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("poly2",d=2, c=1)
        model = SVMNL_Model(2,0.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("poly2",d=2, c=0)
        model = SVMNL_Model(2,1.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("poly2",d=2, c=1)
        model = SVMNL_Model(2,1.0,1,kernel)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)
    
    def test_SVMNL_Poly_Reb(self):
        kernel = Kernel("poly2",d=2, c=0)
        model = SVMNL_Model(2,0.0,1,kernel, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("poly2",d=2, c=1)
        model = SVMNL_Model(2,0.0,1,kernel, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("poly2",d=2, c=0)
        model = SVMNL_Model(2,1.0,1,kernel, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        kernel = Kernel("poly2",d=2, c=1)
        model = SVMNL_Model(2,1.0,1,kernel, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)


    def test_SVM_linear(self):
        model = SVML_Model(2, 1, 0.1)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 1, 1)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 1, 10)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 10, 0.1)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 10, 1)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 10, 10)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)
    
    def test_SVM_linear_Reb(self):
        model = SVML_Model(2, 1, 0.1, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 1, 1, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 1, 10, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 10, 0.1, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 10, 1, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)

        model = SVML_Model(2, 10, 10, rebalance=True)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin,self.LTEbin)
        print(1-acc)
    
    def test_GMM_LBG_ll(self):
        X = np.load("data/GMM_data_4D.npy")
        #print(X)


        model = GMMLBG_Model(3, 1e-6, 2, verbose=True, constrained=False)
        model._train(X,np.zeros(1))

    def test_GMM_LBG_Classify_FC(self):
        model = GMMLBG_Model(3, 1e-6,1, bound= 0.5)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        

        model = GMMLBG_Model(3, 1e-6,2, bound = 0.5)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Model(3, 1e-6,3, bound=0.5)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)



        model = GMMLBG_Model(3, 1e-6,4, bound = 0.5)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

    
    def test_GMM_LBG_Classify_T(self):
        model = GMMLBG_Tied_Model(3, 1e-6,1)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)
        
        model = GMMLBG_Tied_Model(3, 1e-6,2)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Tied_Model(3, 1e-6,3)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Tied_Model(3, 1e-6,4)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)
    def test_GMM_LBG_Classify_D(self):
        model = GMMLBG_Diag_Model(3, 1e-6,1)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)
        
        model = GMMLBG_Diag_Model(3, 1e-6,2)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Diag_Model(3, 1e-6,3)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Diag_Model(3, 1e-6,4)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)
    
    def test_GMM_LBG_Classify_T(self):
        model = GMMLBG_Tied_Model(3, 1e-6,1)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)
        
        model = GMMLBG_Tied_Model(3, 1e-6,2)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Tied_Model(3, 1e-6,3)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)

        model = GMMLBG_Tied_Model(3, 1e-6,4)
        model.train(self.DTR, self.LTR)
        acc, pred, S =model.predict(self.DTE,self.LTE)
        print(1-acc)
        print(S)

    def test_BD_simple(self):
        model = MVG_Model(3, False, False )
        model.train(self.DTR, self.LTR)
        m=model.getConfusionMatrix(self.DTE, self.LTE)
        

        model = MVG_Model(3, False, True )
        model.train(self.DTR, self.LTR)
        m=model.getConfusionMatrix(self.DTE, self.LTE)
        print(m)

        model = MVG_Model(3, True, False )
        model.train(self.DTR, self.LTR)
        m=model.getConfusionMatrix(self.DTE, self.LTE)
        print(m)

        model = MVG_Model(3, True, True )
        model.train(self.DTR, self.LTR)
        m=model.getConfusionMatrix(self.DTE, self.LTE)
        print(m)
    
    def test_CM_comm(self):
        DTR, DTE, LTE, _ = get_Commedia_data()
        
        model_text = Discrete_Model(3, 0.001, prior = vcol(np.ones(3)/3), label_translate=_)

        model_text.train(DTR, _)

        m=model_text.getConfusionMatrix(DTE, LTE)

        print(m)

    def test_BD_Opt(self):
        DTR, DTE, LTE, _ = get_Inf_Par()

        C1 = np.array(
        [[0,1],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.5)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(m)

        C1 = np.array(
        [[0,1],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.8)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(m)

        C1 = np.array(
        [[0,10],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.5)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(m)

        C1 = np.array(
        [[0,1],
        [10,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.8)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(m)

    def test_BD_risks(self):
        DTR, DTE, LTE, _ = get_Inf_Par()

        C1 = np.array(
        [[0,1],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.5)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(f"risk : {b_d1.get_risk(m)}")
        print(f"norm risk : {b_d1.get_norm_risk(m)}")
        print(f"best thresholds: {b_d1.compute_best_threshold(DTE, LTE)}")

        C1 = np.array(
        [[0,1],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.8)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(f"risk : {b_d1.get_risk(m)}")
        print(f"norm risk : {b_d1.get_norm_risk(m)}")
        print(f"best thresholds: {b_d1.compute_best_threshold(DTE, LTE)}")


        C1 = np.array(
        [[0,10],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.5)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(f"risk : {b_d1.get_risk(m)}")
        print(f"norm risk : {b_d1.get_norm_risk(m)}")
        print(f"best thresholds: {b_d1.compute_best_threshold(DTE, LTE)}")


        C1 = np.array(
        [[0,1],
        [10,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.8)
        b_d1.train(DTR, LTE)
        m = b_d1.computeConfusionMatrix(DTE, LTE)
        print(f"risk : {b_d1.get_risk(m)}")
        print(f"norm risk : {b_d1.get_norm_risk(m)}")
        print(f"best thresholds: {b_d1.compute_best_threshold(DTE, LTE)}")
    
    def test_ROC(self):
        DTR, DTE, LTE, _ = get_Inf_Par()

        C1 = np.array(
        [[0,1],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.5)
        b_d1.train(DTR, LTE)
        b_d1.plot_ROC_over_thresholds(DTE,LTE)
        #m = b_d1.computeConfusionMatrix(DTE, LTE)
    
    def test_BE_plot(self):
        DTR, DTE, LTE, _ = get_Inf_Par()

        C1 = np.array(
        [[0,1],
        [1,0]]
        )
        b_d1 = BD_Wrapper("discrete", 2, C1, e_prior=0.5)
        b_d1.train(DTR, LTE)
        b_d1.plot_Bayes_errors(DTE, LTE)
    
    def test_Gauss_Preproc(self):
        p = Gaussianize()

        g,l = p.learn(self.DTR, self.LTR)

        t, tl = p.apply(self.DTE, self.LTE)

        print(t)
        myHistogram(t, 3, tl)
    
    def test_best_pars_LR(self):
        l = 10e-2
        model = LRBinary_Model(2, l)
        kcv = KCV(model, 5)
        
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 0,(-8, 0) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)

    
    def test_best_pars_SVML(self):
        model = SVML_Model(2, 1, 1)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 1,(-4, 4) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)

        model = SVML_Model(2, 1, 0.1)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 0,(0, 1) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)
    
    def test_best_pars_SVMNL_Poly(self):
        kernel = Kernel("poly2",d=2, c=1)
        model = SVMNL_Model(2, 1, 1, kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 1,(-4, 4) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)

        model = SVMNL_Model(2, 1, 0.1, kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 0,(0, 1) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)

    def test_best_pars_SVMNL_RBF(self):
        kernel = Kernel("RBF",gamma=1.0)
        model = SVMNL_Model(2, 1, 1, kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 1,(-4, 4) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)

        model = SVMNL_Model(2, 1, 0.1, kernel)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 0,(0, 1) )
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)
    
    def test_best_pars_bound(self):
        model = GMMLBG_Model(2,1e-6,3, 3, bound=0.01)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 2,(-6, 2))
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)


    def test_best_pars_GMM(self):
        model = GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 0,(1, 6), logbase=2, n_vals = 6, logBounds=False)
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)

        model = GMMLBG_Diag_Model(2,1e-3,3, bound=0.05)
        kcv = KCV(model, 5)
        minDCFs, vals, accs = kcv.find_best_par(model, self.DTRbin, self.LTRbin, 1,(-4, 0))
        minDCFs = [minDCFs]
        accs = [accs]
        
        plot_vals(minDCFs, vals)
        plot_vals(accs, vals)


    def test_load_ds(self):
        D, L = load_ds("PulsarTrain.txt")
        print(D.shape)
        print(L.shape)

        D, L = load_ds("PulsarTest.txt")
        print(D.shape)
        print(L.shape)

        D, L = load_ds("WineTrain.txt")
        print(D.shape)
        print(L.shape)

        D, L = load_ds("WineTest.txt")
        print(D.shape)
        print(L.shape)

        D, L = load_ds("GenderTrain.txt", separator=", ")
        print(D.shape)
        print(L.shape)

        D, L = load_ds("GenderTest.txt")
        print(D.shape)
        print(L.shape)
    
    def test_load_shuffle(self):
        (D, L), (T,TL) = load_Gender(shuffle = True)

        print(L)
        print(TL)

    def test_GAUSS_PCA(self):
        preproc = Gaussianize()
        preproc.addNext(PCA(8))

        DTR, LTR = preproc.learn(self.DTR, self.LTR)
        print(DTR)
        print(LTR)

    def test_quad_LR(self):
        l = 10e-6
        model = QuadLR_Model(2, l, rebalance=False)
        model.train(self.DTRbin, self.LTRbin)
        acc, preds, S = model.predict(self.DTEbin, self.LTEbin)
        print(1-acc)
    
    def test_Znorm(self):
        preproc = Znorm()

        DTR, LTR = preproc.learn(self.DTR, self.LTR)

        print(DTR.mean(1))
        print(DTR.std(1))

        myHistogram(DTR, 2, LTR)

        
        

        

if __name__ == "__main__":

    testClass = LibTest()

    #testClass.test_PCA()
    #testClass.test_LDA()
    #testClass.test_PCA_bin()
    #testClass.test_LDA_bin()
    #testClass.test_PCA_LDA_Combined()
    #testClass.test_MVG_Model_FC()
    #testClass.test_MVG_Model_N()
    #testClass.test_MVG_Model_T()
    #testClass.test_MVG_Model_NT()
    #testClass.test_KCV_FC()
    #testClass.test_KCV_N()
    #testClass.test_KCV_T()
    #testClass.test_KCV_NT()
    #testClass.test_LogReg()
    #testClass.test_LogReg_Rebalanced()
    #testClass.test_quad_LR()
    #testClass.test_SVMNL_RBF()
    #testClass.test_SVMNL_Poly()
    #testClass.test_SVMNL_Poly_Reb()
    #testClass.test_SVM_linear()
    #testClass.test_SVM_linear_Reb()
    #testClass.test_GMM_LBG_ll()
    #testClass.test_GMM_LBG_Classify_FC()
    #testClass.test_GMM_LBG_Classify_D()
    #testClass.test_GMM_LBG_Classify_T()
    #testClass.test_BD_simple()
    #testClass.test_CM_comm()
    #testClass.test_BD_Opt()
    #testClass.test_BD_risks()
    #testClass.test_ROC()
    #testClass.test_BE_plot()
    testClass.test_Gauss_Preproc()
    #testClass.test_best_pars_LR()
    #testClass.test_best_pars_SVML()
    #testClass.test_best_pars_SVMNL_Poly()
    #testClass.test_best_pars_SVMNL_RBF()
    #testClass.test_best_pars_GMM()
    #testClass.test_load_ds()
    #testClass.test_load_shuffle()
    #testClass.test_GAUSS_PCA()
    #testClass.test_Znorm()

    



