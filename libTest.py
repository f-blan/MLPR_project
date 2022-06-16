from src_lib import *
from data import *
import unittest

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
        acc=kcv.crossValidate(self.D, self.L)
        print(1-acc)

    def test_KCV_N(self):
        model = MVG_Model(3, False, True)
        kcv = KCV(model, -1, LOO = True)
        acc=kcv.crossValidate(self.D, self.L)
        print(1-acc)
    
    def test_KCV_T(self):
        model = MVG_Model(3, True, False)
        kcv = KCV(model, -1, LOO = True)
        acc=kcv.crossValidate(self.D, self.L)
        print(1-acc)

    def test_KCV_NT(self):
        model = MVG_Model(3, True, True)
        kcv = KCV(model, -1, LOO = True)
        acc=kcv.crossValidate(self.D, self.L)
        print(1-acc)

        

        

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
    testClass.test_KCV_FC()
    testClass.test_KCV_N()
    testClass.test_KCV_T()
    testClass.test_KCV_NT()



