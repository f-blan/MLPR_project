from src_lib import *
from data import *
from src_lib.GMM_Model import GMMLBG_DT_Model, GMMLBG_Diag_Model, GMMLBG_Model, GMMLBG_Tied_Model
from src_lib.SVM_Model import Kernel

FOLDS = 5
VERBOSE = True
STOP_TH = 1e-3

class ExperimentsCal:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))

        self.model1 = GMMLBG_Tied_Model(2,STOP_TH, 4)

        preproc = PCA(8)
        self.model2 = GMMLBG_Model(2,STOP_TH, 3, preProcess=preproc)
    
    def find_act_DCF_primary(self):

        kcv = KCV(self.model1, 5)
        actDCFb, thb = kcv.compute_actual_dcf(self.model1, self.DTR, self.LTR, self.bal_app[0])

        kcv = KCV(self.model1, 5)
        actDCFf, thf = kcv.compute_actual_dcf(self.model1, self.DTR, self.LTR, self.female_app[0])
        
        kcv = KCV(self.model1, 5)
        actDCFm, thm = kcv.compute_actual_dcf(self.model1, self.DTR, self.LTR, self.male_app[0])


        if VERBOSE:
            print(f"act DCF for primary model bal: {actDCFb} - {thb}")
            print(f"act DCF for primary model female: {actDCFf} - {thf}")
            print(f"act DCF for primary model male: {actDCFm} - {thm}")
    
    def find_act_DCF_secondary(self):
        
        kcv = KCV(self.model2, 5)
        actDCFb, thb = kcv.compute_actual_dcf(self.model2, self.DTR, self.LTR, self.bal_app[0])
        
        kcv = KCV(self.model2, 5)
        actDCFf, thf = kcv.compute_actual_dcf(self.model2, self.DTR, self.LTR, self.female_app[0])
        
        kcv = KCV(self.model2, 5)
        actDCFm, thm = kcv.compute_actual_dcf(self.model2, self.DTR, self.LTR, self.male_app[0])


        if VERBOSE:
            print(f"act DCF for secondary model bal: {actDCFb} - {thb}")
            print(f"act DCF for secondary model female: {actDCFf} - {thf}")
            print(f"act DCF for secondary model male: {actDCFm} - {thm}")
    
    def Bayes_plot(self):
        DCFsList = []
        kcv = KCV(self.model1, 5)
        lOdds, DCFsP, minDCFsP = kcv.compute_bayes_pars(self.model1, self.DTR, self.LTR )

        DCFsList.append(DCFsP)
        DCFsList.append(minDCFsP)
        
        kcv = KCV(self.model2, 5)
        lOdds, DCFsS, minDCFsS = kcv.compute_bayes_pars(self.model2, self.DTR, self.LTR )

        DCFsList.append(DCFsS)
        DCFsList.append(minDCFsS)

        if VERBOSE:
            plot_vals(DCFsList, lOdds,False)

    def threshold_estimate_primary(self):
        DCFsList = []

        kcv = KCV(self.model1, 5)
        minDCFb, theory_actDCFb, estimate_th_actDCFb, best_thb = kcv.threshold_estimate(self.model1, self.DTR, self.LTR, self.bal_app[0])
        

        kcv = KCV(self.model1, 5)
        minDCFf, theory_actDCFf, estimate_th_actDCFf, best_thf = kcv.threshold_estimate(self.model1, self.DTR, self.LTR, self.female_app[0])
        
        kcv = KCV(self.model1, 5)
        kcv = KCV(self.model1, 5)
        minDCFm, theory_actDCFm, estimate_th_actDCFm, best_thm = kcv.threshold_estimate(self.model1, self.DTR, self.LTR, self.male_app[0])
        

        if VERBOSE:
            print(f"bal, minDCF vs estimated th actDCF : {minDCFb} - {theory_actDCFb} - {estimate_th_actDCFb}. Best th: {best_thb}")
            print(f"female, minDCF vs estimated th actDCF : {minDCFf} - {theory_actDCFf} - {estimate_th_actDCFf}. Best th: {best_thf}")
            print(f"male, minDCF vs estimated th actDCF : {minDCFm} - {theory_actDCFm} - {estimate_th_actDCFm}. Best th: {best_thm}")
    
    def threshold_estimate_secondary(self):
        kcv = KCV(self.model2, 5)
        minDCFb, theory_actDCFb, estimate_th_actDCFb, best_thb = kcv.threshold_estimate(self.model2, self.DTR, self.LTR, self.bal_app[0])
        

        kcv = KCV(self.model2, 5)
        minDCFf, theory_actDCFf, estimate_th_actDCFf, best_thf = kcv.threshold_estimate(self.model2, self.DTR, self.LTR, self.female_app[0])
        
        kcv = KCV(self.model2, 5)
        minDCFm, theory_actDCFm, estimate_th_actDCFm, best_thm = kcv.threshold_estimate(self.model2, self.DTR, self.LTR, self.male_app[0])
        

        if VERBOSE:
            print(f"bal, minDCF vs estimated th actDCF : {minDCFb} - {theory_actDCFb} - {estimate_th_actDCFb}. Best th: {best_thb}")
            print(f"female, minDCF vs estimated th actDCF : {minDCFf} - {theory_actDCFf} - {estimate_th_actDCFf}. Best th: {best_thf}")
            print(f"male, minDCF vs estimated th actDCF : {minDCFm} - {theory_actDCFm} - {estimate_th_actDCFm}. Best th: {best_thm}")

    def calibration_primary(self):
        kcv = KCV(self.model1, 5)
        actDCFbB, actDCFfB, actDCFmB = kcv.calibrator_eval(self.model1, self.DTR, self.LTR, self.bal_app[0])

        kcv = KCV(self.model1, 5)
        actDCFbF, actDCFfF, actDCFmF = kcv.calibrator_eval(self.model1, self.DTR, self.LTR, self.female_app[0])

        kcv = KCV(self.model1, 5)
        actDCFbM, actDCFfM, actDCFmM = kcv.calibrator_eval(self.model1, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"result with logReg pi = 0.5 for varying applications: {actDCFbB} - {actDCFfB} - {actDCFmB}")
            print(f"result with logReg pi = 0.9 for varying applications: {actDCFbF} - {actDCFfF} - {actDCFmF}")
            print(f"result with logReg pi = 0.1 for varying applications: {actDCFbM} - {actDCFfM} - {actDCFmM}")
    
    def Bayes_plot_cal(self):
        DCFsList = []
        kcv = KCV(self.model1, 5)
        lOdds, calDCFs, uncDCFs, minDCFs = kcv.compute_calibrated_bayes_pars(self.model1, self.DTR, self.LTR )

        DCFsList.append(calDCFs)
        DCFsList.append(minDCFs)
        DCFsList.append(uncDCFs)
        
        """
        kcv = KCV(self.model2, 5)
        lOdds, DCFsS, minDCFsS = kcv.compute_calibrated_bayes_pars(self.model2, self.DTR, self.LTR )

        DCFsList.append(DCFsS)
        DCFsList.append(minDCFsS)
        """
        if VERBOSE:
            plot_vals(DCFsList, lOdds,False, compare_mode=False)



  

if __name__ == "__main__":
    exps = ExperimentsCal("gend")

    #exps.find_act_DCF_primary()
    #exps.find_act_DCF_secondary()
    #exps.Bayes_plot()
    #exps.threshold_estimate_primary()
    #exps.threshold_estimate_secondary()
    #exps.calibration_primary()
    exps.Bayes_plot_cal()
        