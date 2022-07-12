from re import M
from src_lib import *
from data import *
from src_lib.GMM_Model import GMMLBG_DT_Model, GMMLBG_Diag_Model, GMMLBG_Model, GMMLBG_Tied_Model
from src_lib.SVM_Model import Kernel
from src_lib.Fusion_Model import Fusion_Model


FOLDS = 5
VERBOSE = True
STOP_TH = 1e-3

class ExperimentsTest:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))

        preproc = PCA(8)
        self.GMM_T = GMMLBG_Tied_Model(2,STOP_TH, 4)
        self.GMM_T_PCA = GMMLBG_Tied_Model(2,STOP_TH, 4, preProcess=preproc)
        

        preproc = PCA(8)
        self.GMM_FC_PCA = GMMLBG_Model(2,STOP_TH, 3, preProcess=preproc)
        self.GMM_FC = GMMLBG_Model(2, STOP_TH, 3)

        kcv = KCV(None, 8)
        self.fusion = Fusion_Model(2,kcv, calibrate_after=True)
        self.fusion.mode = "fullTraining"
        self.fusion.add_model(self.GMM_T, False)
        self.fusion.add_model(self.GMM_FC, False)

        preproc = PCA(8)
        kernel = Kernel(kname="RBF", gamma = 0.01)
        self.SVM = SVMNL_Model(2, 0, 20, kernel=kernel, preProcess=preproc)

        self.SVML = SVML_Model(2, 1, 1)

        preproc = PCA(8)
        self.LR = LRBinary_Model(2, 0.001)
        self.LR_PCA = LRBinary_Model(2, 0.001,preProcess=preproc)
        preproc = Gaussianize()
        self.LR_Gauss = LRBinary_Model(2, 0.001, preProcess=preproc)

        preproc = PCA(8)
        self.QuadLR = QuadLR_Model(2, 10e4)
        self.QuadLR_PCA = QuadLR_Model(2, 10e4,preProcess=preproc)
        preproc = Gaussianize()
        self.QuadLR_Gauss = QuadLR_Model(2, 10e-2, preProcess=preproc)

        preproc = PCA(8)
        self.MVGFC  = MVG_Model(2,False,False)
        self.MVGFC_PCA = MVG_Model(2, False, False, preProcess=preproc)
        preproc = Gaussianize()
        self.MVGFC_Gauss = MVG_Model(2, False, False, preProcess=preproc)

        preproc = PCA(8)
        self.MVGT= MVG_Model(2, True, False)
        self.MVGT_PCA = MVG_Model(2, True, False, preProcess=preproc)
        preproc = Gaussianize()
        self.MVGT_Gauss = MVG_Model(2, True, False, preProcess=preproc)
    
    def _train_fully(self, model: Model):
        model.train(self.DTR, self.LTR)
    
    def _predict_test_scores(self, model: Model):
        _,_,S= model.predict(self.DTE, self.LTE)
        return S
    
    def _compute_minDCF_all_apps(self, model: Model):
        model.train(self.DTR, self.LTR)
        S = self._predict_test_scores(model)

        w = BD_Wrapper(model, 2, e_prior=self.bal_app[0])
        minDCFb, _ = w.compute_best_threshold_from_Scores(S, self.LTE)

        w = BD_Wrapper(model, 2, e_prior=self.female_app[0])
        minDCFf, _ = w.compute_best_threshold_from_Scores(S, self.LTE)

        w = BD_Wrapper(model, 2, e_prior=self.male_app[0])
        minDCFm, _ = w.compute_best_threshold_from_Scores(S, self.LTE)

        return minDCFb, minDCFf, minDCFm
    
    def _dcf_over_different_pars(self, model: Model, change_function, iterations: int):
        minDCFsB = []
        minDCFsF = []
        minDCFsM = []

        for i in range(0, iterations):
            change_function(i, model)
            b,f,m = self._compute_minDCF_all_apps(model)
            minDCFsB.append(b)
            minDCFsF.append(f)
            minDCFsM.append(m)

        return [minDCFsB, minDCFsF, minDCFsM]
    
    def test_MVGFC(self):
        b, f, m = self._compute_minDCF_all_apps(self.MVGFC)

        if VERBOSE:
            print("raw")
            print(f"minDCF for MVG FC bal app: {b}")
            print(f"minDCF for MVG FC female app: {f}")
            print(f"minDCF for MVG FC male app: {m}")
        
        b, f, m = self._compute_minDCF_all_apps(self.MVGFC_PCA)

        if VERBOSE:
            print("with PCA")
            print(f"minDCF for MVG FC bal app: {b}")
            print(f"minDCF for MVG FC female app: {f}")
            print(f"minDCF for MVG FC male app: {m}")

        b, f, m = self._compute_minDCF_all_apps(self.MVGFC_Gauss)

        if VERBOSE:
            print("with Gauss")
            print(f"minDCF for MVG FC bal app: {b}")
            print(f"minDCF for MVG FC female app: {f}")
            print(f"minDCF for MVG FC male app: {m}")
    
    def test_MVGT(self):
        b, f, m = self._compute_minDCF_all_apps(self.MVGT)

        if VERBOSE:
            print("raw")
            print(f"minDCF for MVG tied bal app: {b}")
            print(f"minDCF for MVG tied female app: {f}")
            print(f"minDCF for MVG tied male app: {m}")

        b, f, m = self._compute_minDCF_all_apps(self.MVGT_PCA)

        if VERBOSE:
            print("with PCA")
            print(f"minDCF for MVG tied bal app: {b}")
            print(f"minDCF for MVG tied female app: {f}")
            print(f"minDCF for MVG tied male app: {m}")
        
        b, f, m = self._compute_minDCF_all_apps(self.MVGT_Gauss)

        if VERBOSE:
            print("with gauss")
            print(f"minDCF for MVG tied bal app: {b}")
            print(f"minDCF for MVG tied female app: {f}")
            print(f"minDCF for MVG tied male app: {m}")
    
    def test_LR(self):
        par_vals = np.logspace(-3, 3, num=10, base=10)
        def change_fun(x: int, model: LRBinary_Model):
            model.reg_lambda = par_vals[x]

        b, f, m = self._compute_minDCF_all_apps(self.LR)
        DCFList = self._dcf_over_different_pars(self.LR, change_fun, 10)
        
        if VERBOSE:
            print("raw")
            print(f"minDCF for linear LR bal app: {b}")
            print(f"minDCF for linear LR female app: {f}")
            print(f"minDCF for linear LR male app: {m}")
            plot_vals(DCFList, par_vals)
        
        
        b, f, m = self._compute_minDCF_all_apps(self.LR_PCA)
        DCFList = self._dcf_over_different_pars(self.LR_PCA, change_fun, 10)

        if VERBOSE:
            print("pca")
            print(f"minDCF for linear LR bal app: {b}")
            print(f"minDCF for linear LR female app: {f}")
            print(f"minDCF for linear LR male app: {m}")
            plot_vals(DCFList, par_vals)
        
        b, f, m = self._compute_minDCF_all_apps(self.LR_Gauss)
        DCFList = self._dcf_over_different_pars(self.LR_Gauss, change_fun, 10)

        if VERBOSE:
            print("Gauss")
            print(f"minDCF for linear LR bal app: {b}")
            print(f"minDCF for linear LR female app: {f}")
            print(f"minDCF for linear LR male app: {m}")
            plot_vals(DCFList, par_vals)
    
    def test_QuadLR(self):
        par_vals = np.logspace(-3, 6, num=10, base=10)
        def change_fun(x: int, model: QuadLR_Model):
            model.reg_lambda = par_vals[x]
        
        b, f, m = self._compute_minDCF_all_apps(self.QuadLR)
        DCFList = self._dcf_over_different_pars(self.QuadLR, change_fun, 10)
        
        if VERBOSE:
            print("raw")
            print(f"minDCF for linear LR bal app: {b}")
            print(f"minDCF for linear LR female app: {f}")
            print(f"minDCF for linear LR male app: {m}")
            plot_vals(DCFList, par_vals)
        
        
        b, f, m = self._compute_minDCF_all_apps(self.QuadLR_PCA)
        DCFList = self._dcf_over_different_pars(self.QuadLR_PCA, change_fun, 10)

        if VERBOSE:
            print("pca")
            print(f"minDCF for linear LR bal app: {b}")
            print(f"minDCF for linear LR female app: {f}")
            print(f"minDCF for linear LR male app: {m}")
            plot_vals(DCFList, par_vals)
        
        b, f, m = self._compute_minDCF_all_apps(self.QuadLR_Gauss)
        DCFList = self._dcf_over_different_pars(self.QuadLR_Gauss, change_fun, 10)

        if VERBOSE:
            print("Gauss")
            print(f"minDCF for linear LR bal app: {b}")
            print(f"minDCF for linear LR female app: {f}")
            print(f"minDCF for linear LR male app: {m}")
            plot_vals(DCFList, par_vals)
    
    def test_SVMNL(self):
        par_vals = np.logspace(-3, 3, num=5, base=10)
        def change_fun(x: int, model: SVMNL_Model):
            model.C = par_vals[x]
        
        b, f, m = self._compute_minDCF_all_apps(self.SVM)
        DCFList = self._dcf_over_different_pars(self.SVM, change_fun, 5)
        
        if VERBOSE:
            print("raw")
            print(f"minDCF for RBF SVM bal app: {b}")
            print(f"minDCF for RBF SVM female app: {f}")
            print(f"minDCF for RBF SVM male app: {m}")
            plot_vals(DCFList, par_vals)

    def test_SVML(self):
        par_vals = np.logspace(-3, 3, num=5, base=10)
        def change_fun(x: int, model: SVML_Model):
            model.C = par_vals[x]
        
        b, f, m = self._compute_minDCF_all_apps(self.SVML)
        DCFList = self._dcf_over_different_pars(self.SVML, change_fun, 5)
        
        if VERBOSE:
            print("raw")
            print(f"minDCF for linear SVM bal app: {b}")
            print(f"minDCF for linear SVM female app: {f}")
            print(f"minDCF for linear SVM male app: {m}")
            plot_vals(DCFList, par_vals)
    
    def test_GMM_FC(self):
        par_vals = [1,2,3,4]
        def change_fun(x: int, model: GMMLBG_Model):
            model.n_gauss_exp = par_vals[x]
        
        b, f, m = self._compute_minDCF_all_apps(self.GMM_FC)
        DCFList = self._dcf_over_different_pars(self.GMM_FC, change_fun, 4)

        if VERBOSE:
            print("raw")
            print(f"minDCF for GMM FC bal app: {b}")
            print(f"minDCF for GMM FC female app: {f}")
            print(f"minDCF for GMM FC male app: {m}")
            plot_vals(DCFList, par_vals)
        
        b, f, m = self._compute_minDCF_all_apps(self.GMM_FC_PCA)
        DCFList = self._dcf_over_different_pars(self.GMM_FC_PCA, change_fun, 4)

        if VERBOSE:
            print("raw")
            print(f"minDCF for GMM_FC bal app: {b}")
            print(f"minDCF for GMM_FC female app: {f}")
            print(f"minDCF for GMM_FC male app: {m}")
            plot_vals(DCFList, par_vals)

    def test_GMM_T(self):
        par_vals = [1,2,3,4]
        def change_fun(x: int, model: GMMLBG_Tied_Model):
            model.n_gauss_exp = par_vals[x]
        
        b, f, m = self._compute_minDCF_all_apps(self.GMM_T)
        DCFList = self._dcf_over_different_pars(self.GMM_T, change_fun, 4)

        if VERBOSE:
            print("raw")
            print(f"minDCF for GMM T bal app: {b}")
            print(f"minDCF for GMM T female app: {f}")
            print(f"minDCF for GMM T male app: {m}")
            plot_vals(DCFList, par_vals)
        
        b, f, m = self._compute_minDCF_all_apps(self.GMM_T_PCA)
        DCFList = self._dcf_over_different_pars(self.GMM_T_PCA, change_fun, 4)

        if VERBOSE:
            print("raw")
            print(f"minDCF for GMM T bal app: {b}")
            print(f"minDCF for GMM T female app: {f}")
            print(f"minDCF for GMM T male app: {m}")
            plot_vals(DCFList, par_vals)
    
    def test_Fusion(self):
        b, f, m = self._compute_minDCF_all_apps(self.fusion)

        if VERBOSE:
            print(f"minDCF for linear SVM bal app: {b}")
            print(f"minDCF for linear SVM female app: {f}")
            print(f"minDCF for linear SVM male app: {m}")
    
    def _threshold_find(self, model : Model):
        model.train(self.DTR, self.LTR)
        kcv = KCV(model, 5)
        _, S = kcv.crossValidate(self.DTR, self.LTR)

        w = BD_Wrapper(model, 2, e_prior=self.bal_app[0])
        b, thB = w.compute_best_threshold_from_Scores(S, self.LTR)

        w = BD_Wrapper(model, 2, e_prior=self.female_app[0])
        f, thF = w.compute_best_threshold_from_Scores(S, self.LTR)

        w = BD_Wrapper(model, 2, e_prior=self.male_app[0])
        m, thM = w.compute_best_threshold_from_Scores(S, self.LTR)

        return  thB, thF, thM
    
    def _opt_th_actDCF_compute(self, model: Model, thb: float, thf: float, thm: float):
        model.train(self.DTR, self.LTR)
        _, __, S =model.predict(self.DTE, self.LTE)

        w = BD_Wrapper(model, 2, e_prior=self.bal_app[0])
        m = w.get_matrix_from_threshold(self.LTE, S, thb)
        actB = w.get_norm_risk(m)

        w = BD_Wrapper(model, 2, e_prior=self.female_app[0])
        m = w.get_matrix_from_threshold(self.LTE, S, thf)
        actF = w.get_norm_risk(m)

        w = BD_Wrapper(model, 2, e_prior=self.male_app[0])
        m = w.get_matrix_from_threshold(self.LTE, S, thm)
        actM = w.get_norm_risk(m)

        return actB, actF, actM

    def threshold_actDCF(self):
        thb, thf, thm = self._threshold_find(self.GMM_FC)
        b, f, m = self._opt_th_actDCF_compute(self.GMM_FC, thb, thf, thm)

        if VERBOSE:
            print(f"act DCF for GMM FC bal: {b}")
            print(f"act DCF for GMM FC female: {f}")
            print(f"act DCF for GMM FC male: {m}")

        thb, thf, thm = self._threshold_find(self.GMM_T)
        b, f, m = self._opt_th_actDCF_compute(self.GMM_T, thb, thf, thm)

        if VERBOSE:
            print(f"act DCF for GMM T bal: {b}")
            print(f"act DCF for GMM T female: {f}")
            print(f"act DCF for GMM T male: {m}")
    
    def calibration_actDCF(self):
        model = self.GMM_T
    
        calScores = self._get_Calibrated_test_Scores(model)

        w = BD_Wrapper(model, 2, e_prior=self.bal_app[0])
        th = w.get_theoretical_threshold()
        m = w.get_matrix_from_threshold(self.LTE, calScores, th)
        actB = w.get_norm_risk(m)

        w = BD_Wrapper(model, 2, e_prior=self.female_app[0])
        th = w.get_theoretical_threshold()
        m = w.get_matrix_from_threshold(self.LTE, calScores, th)
        actF = w.get_norm_risk(m)

        w = BD_Wrapper(model, 2, e_prior=self.male_app[0])
        th = w.get_theoretical_threshold()
        m = w.get_matrix_from_threshold(self.LTE, calScores, th)
        actM = w.get_norm_risk(m)

        if VERBOSE:
            print(f"act DCF for calibrated GMM T bal : {actB}")
            print(f"act DCF for calibrated GMM T bal : {actF}")
            print(f"act DCF for calibrated GMM T bal : {actM}")

    def _get_Calibrated_test_Scores(self, model: Model):
        cw = C_Wrapper(e_prior=0.5)
        kcv = KCV(model, FOLDS)
        _, S = kcv.crossValidate(self.DTR, self.LTR)

        calibrator = cw.calibrator
        calibrator.train(vrow(S), self.LTR)
        
        model.train(self.DTR, self.LTR)
        _, _, ncScores = model.predict(self.DTE, self.LTE)
        _, __, calScores = calibrator.predict(ncScores, self.LTE)
        return calScores

    def fusion_actDCF(self):
        w= BD_Wrapper("static", 2)
        self.fusion.mode = "fullTrain"
        self.fusion.train(self.DTR, self.LTR)
        _, __, scoreF = self.fusion.predict(self.DTE, self.LTE)

        w= BD_Wrapper("static", 2, e_prior=self.bal_app[0])
        th = w.get_theoretical_threshold()
        mb =w.get_matrix_from_threshold(self.LTE, scoreF, th)
        b = w.get_norm_risk(mb)

        w= BD_Wrapper("static", 2, e_prior=self.female_app[0])
        th = w.get_theoretical_threshold()
        mf =w.get_matrix_from_threshold(self.LTE, scoreF, th)
        f = w.get_norm_risk(mf)

        w= BD_Wrapper("static", 2, e_prior=self.male_app[0])
        th = w.get_theoretical_threshold()
        mm =w.get_matrix_from_threshold(self.LTE, scoreF, th)
        m = w.get_norm_risk(mm)

        if VERBOSE:
            print(f"act DCF for fusion bal : {b} - {(mb[1,1]+mb[0,0])/mb.sum()}")
            print(f"act DCF for fusion bal : {f} - {(mf[1,1]+mf[0,0])/mf.sum()}")
            print(f"act DCF for fusion bal : {m} - {(mm[1,1]+mm[0,0])/mm.sum()}")

    def compare_BPs_best_Models(self):
        DCFList = []
        
        w= BD_Wrapper("static", 2)
        scoreT = self._get_Calibrated_test_Scores(self.GMM_T)
        priorLogOdds, actT, minT = w.plot_Bayes_errors_from_scores(scoreT, self.LTE, plot = False)
        DCFList.append(minT)
        DCFList.append(actT)

        w= BD_Wrapper("static", 2)
        self.GMM_FC.train(self.DTR, self.LTR)
        scoreFC = self._predict_test_scores(self.GMM_FC)
        priorLogOdds, actFC, minFC = w.plot_Bayes_errors_from_scores(scoreFC, self.LTE, plot = False)
        DCFList.append(minFC)
        DCFList.append(actFC)

        w= BD_Wrapper("static", 2)
        self.fusion.mode = "fullTrain"
        self.fusion.train(self.DTR, self.LTR)
        _, __, scoreF = self.fusion.predict(self.DTE, self.LTE)
        priorLogOdds, actT, minT = w.plot_Bayes_errors_from_scores(scoreF, self.LTE, plot = False)
        DCFList.append(minT)
        DCFList.append(actT)

        if VERBOSE:
            plot_vals(DCFList, priorLogOdds,False, compare_mode=True)


if __name__ == "__main__":
    exps = ExperimentsTest("gend")

    #exps.test_MVGFC()
    #exps.test_MVGT()
    #exps.test_LR()
    #exps.test_QuadLR()
    #exps.test_SVMNL()
    #exps.test_SVML()
    #exps.test_GMM_FC()
    #exps.test_GMM_T()
    #exps.test_Fusion()
    #exps.threshold_actDCF()
    #exps.calibration_actDCF()
    #exps.fusion_actDCF()
    exps.compare_BPs_best_Models()




    



