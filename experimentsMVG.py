from src_lib import *
from data import *

FOLDS = 5
VERBOSE = True

class ExperimentsMVG:
    def __init__(self, dataName: str):
        if dataName == "gend":
            (self.DTR, self.LTR), (self.DTE, self.LTE) = load_Gender(shuffle=True)
        
        self.bal_app = (0.5, np.array([[1,0],[0,1]]))
        self.female_app = (0.9, np.array([[1,0],[0,1]]))
        self.male_app = (0.1, np.array([[1,0],[0,1]]))
        
    
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
    
    def MVG_FC_Z(self):
        preproc = Znorm()
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_GAUSS: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_GAUSS: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_GAUSS: {minDCFm}, th: {_m}")
            #last res: 0.049, 0.1206, 0.1266
    
    def MVG_FC_GAUSS(self):
        preproc = Gaussianize()
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_GAUSS: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_GAUSS: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_GAUSS: {minDCFm}, th: {_m}")
            #last res: 0.0613 - 0.1773 - 0.1856 
    
    def MVG_FC_PCA_10(self):
        preproc = PCA(10)
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_PCA10: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_PCA10: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_PCA10: {minDCFm}, th: {_m}")
            #last res: 0.0473 - 0.1183 - 0.1393
    
    def MVG_FC_Z_PCA_10(self):
        preproc = Znorm()
        preproc.addNext( PCA(10))
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_Z_PCA10: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_Z_PCA10: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_Z_PCA10: {minDCFm}, th: {_m}")

    def MVG_FC_PCA_8(self):
        preproc = PCA(8)
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.0446 - 0.1226 - 0.1403
    
    def MVG_FC_Z_PCA_8(self):
        preproc = Znorm()
        preproc.addNext( PCA(8))
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_Z_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_Z_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_Z_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.1770 - 0.4210 - 0.4493
    
    def MVG_FC_GAUSS_PCA_8(self):
        preproc = Gaussianize()
        preproc.addNext(PCA(8))
        
        model = MVG_Model(2, False, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_GAUSS_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_GAUSS_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_GAUSS_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.1663 - 0.4233 - 0.4380
            
    
    def MVG_N_Raw(self):
        model = MVG_Model(2, False, True)
        kcvw = KCV(model, FOLDS)
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_N_RAW: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_N_RAW: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_N_RAW: {minDCFm}, th: {_m}")
            #last res: 0.5654 - 0.8616 - 0.8170
    
    def MVG_N_GAUSS(self):
        preproc = Gaussianize()
        model = MVG_Model(2, False, True, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_N_GAUSS: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_N_GAUSS: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_N_GAUSS: {minDCFm}, th: {_m}")
            #last res: 0.5409 - 0.8356 - 0.8029
    
    def MVG_N_Z(self):
        preproc = Znorm()
        model = MVG_Model(2, False, True, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_N_Z: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_N_Z: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_N_GAUSS: {minDCFm}, th: {_m}")
            #last res: 0.5653 - 0.8616 - 0.8170

    def MVG_N_PCA_10(self):
        preproc = PCA(10)
        model = MVG_Model(2, False, True, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_N_PCA10: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_N_PCA10: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_N_PCA10: {minDCFm}, th: {_m}")
            #last res: 0.0670 - 0.1646 - 0.1663
            
    
    def MVG_N_PCA_8(self):
        preproc = PCA(8)
        model = MVG_Model(2,False, True, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_N_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_N_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_N_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.0673 - 0.1623 - 0.1703
    
    def MVG_N_Z_PCA_8(self):
        preproc = Znorm()
        preproc.addNext( PCA(8))
        model = MVG_Model(2, False, True, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_Z_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_Z_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_FC_Z_PCA8: {minDCFm}, th: {_m}")
            

    def MVG_N_GAUSS_PCA_8(self):
        preproc = Gaussianize()
        preproc.addNext(PCA(8))
        
        model = MVG_Model(2, False, True, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_N_GAUSS_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_N_GAUSS_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_N_GAUSS_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.1720 - 0.4240 - 0.4486
        
    
    def MVG_T_Raw(self):
        model = MVG_Model(2, True, False)
        kcvw = KCV(model, FOLDS)
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_T_RAW: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_T_RAW: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_T_RAW: {minDCFm}, th: {_m}")
            #last res: 0.047 - 0.1263 - 0.1210
            
    
    def MVG_T_GAUSS(self):
        preproc = Gaussianize()
        model = MVG_Model(2, True, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)
        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])
        
        if VERBOSE:
            print(f"MINDCF for MVG_T_GAUSS: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_T_GAUSS: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_T_GAUSS: {minDCFm}, th: {_m}")
            #last res: 0.0593 - 0.1646 - 0.1790
    
    def MVG_T_Z(self):
        preproc = Znorm()
        model = MVG_Model(2, True, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)
        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])
        
        if VERBOSE:
            print(f"MINDCF for MVG_T_GAUSS: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_T_GAUSS: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_T_GAUSS: {minDCFm}, th: {_m}")
            
    
    def MVG_T_PCA_10(self):
        preproc = PCA(10)
        model = MVG_Model(2, True, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_T_PCA10: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_T_PCA10: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_T_PCA10: {minDCFm}, th: {_m}")
            #last res: 0.0470 - 0.1243 - 0.1303 
    
    def MVG_T_PCA_8(self):
        preproc = PCA(8)
        model = MVG_Model(2,True, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_T_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_T_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_T_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.0446 - 0.1256 - 0.132

    def MVG_T_Z_PCA_8(self):
        preproc = Znorm()
        preproc.addNext( PCA(8))
        model = MVG_Model(2,  True, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_FC_Z_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_FC_Z_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for female MVG_FC_Z_PCA8: {minDCFm}, th: {_f}")

    def MVG_T_GAUSS_PCA_8(self):
        preproc = Gaussianize()
        preproc.addNext(PCA(8))
        
        model = MVG_Model(2, True, False, preProcess=preproc)
        kcvw = KCV(model, FOLDS)

        
        minDCF, _ = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.bal_app[0])
        minDCFf, _f = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.female_app[0])
        minDCFm, _m = kcvw.compute_min_dcf(model, self.DTR, self.LTR, self.male_app[0])

        if VERBOSE:
            print(f"MINDCF for MVG_T_GAUSS_PCA8: {minDCF}, th: {_}")
            print(f"MINDCF for female MVG_T_GAUSS_PCA8: {minDCFf}, th: {_f}")
            print(f"MINDCF for male MVG_T_GAUSS_PCA8: {minDCFm}, th: {_m}")
            #last res: 0.1643 - 0.4076 - 0.4323
    
    def random_test(self):
        preproc1 = PCA(8)
        preproc1.addNext(Znorm())

        preproc2 = PCA(8)

        model1 = MVG_Model(2, False,False, preProcess=preproc1)
        model2 = MVG_Model(2, False,False, preProcess=preproc2)

        model1.train(self.DTR, self.LTR)
        model2.train(self.DTR, self.LTR)

        print(model1.predict(self.DTE, self.LTE)[0])
        
        print(model2.predict(self.DTE, self.LTE)[0])







if __name__ == "__main__":
    exps = ExperimentsMVG("gend")
    
    #exps.MVG_FC_Raw()
    #exps.MVG_FC_Z()
    #exps.MVG_FC_GAUSS()
    #exps.MVG_FC_PCA_10()
    #exps.MVG_FC_PCA_8()
    #exps.MVG_FC_GAUSS_PCA_8()
    #exps.MVG_FC_Z_PCA_8()
    #exps.MVG_FC_Z_PCA_10()
    #exps.MVG_N_Raw()
    #exps.MVG_N_Z()
    #exps.MVG_N_GAUSS()
    #exps.MVG_N_PCA_10()
    #exps.MVG_N_PCA_8()
    #exps.MVG_N_Z_PCA_8()
    #exps.MVG_N_GAUSS_PCA_8()
    #exps.MVG_T_Raw()
    #exps.MVG_T_GAUSS()
    #exps.MVG_T_PCA_10()
    #exps.MVG_T_PCA_8()
    #exps.MVG_T_GAUSS_PCA_8()
    #exps.MVG_T_Z()
    #exps.MVG_T_Z_PCA_8()
    #exps.random_test()
        