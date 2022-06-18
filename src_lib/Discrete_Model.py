import numpy as np
import itertools

from data.data_loader import *

from src_lib import *
"""
    NOTE: this file is not being used for the project itself but it's here just for testing purposes


"""


def compute_accuracy(P, L):

    '''
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    '''

    PredictedLabel = np.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect)/float(NTotal), PredictedLabel

def mcol(v):
    return v.reshape((v.size, 1))

def tercet2occurrencies(tercet, hWordDict):
    
    '''
    Convert a tercet in a (column) vector of word occurrencies. Word indices are given by hWordDict
    '''
    v = np.zeros(len(hWordDict))
    for w in tercet.split():
        if w in hWordDict: # We discard words that are not in the dictionary
            v[hWordDict[w]] += 1
    return mcol(v)


def compute_logLikelihoodMatrix(h_clsLogProb, hWordDict, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb and hWordDict are the dictionary of model parameters and word indices as returned by S2_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}
    
    numClasses = len(h_clsLogProb)
    numWords = len(hWordDict)

    # We build the matrix of model parameters. Each row contains the model parameters for a class (the row index is given from hCls2Idx)
    MParameters = np.zeros((numClasses, numWords)) 
    for cls in h_clsLogProb:
        clsIdx = hCls2Idx[cls]
        MParameters[clsIdx, :] = h_clsLogProb[cls] # MParameters[clsIdx, :] is a 1-dimensional view that corresponds to the row clsIdx, we can assign to the row directly the values of another 1-dimensional array

    SList = []
    for tercet in lTercets:
        v = tercet2occurrencies(tercet, hWordDict)
        STercet = np.dot(MParameters, v) # The log-lieklihoods for the tercets can be computed as a matrix-vector product. Each row of the resulting column vector corresponds to M_c v = sum_j v_j log p_c,j
        SList.append(np.dot(MParameters, v))

    S = np.hstack(SList)
    return S


def compute_classPosteriors(S, logPrior = None):

    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = np.log( np.ones(S.shape[0]) / float(S.shape[0]) )
    J = S + mcol(logPrior) # Compute joint probability
    ll = sp.special.logsumexp(J, axis = 0) # Compute marginal likelihood log f(x)
    P = J - ll # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return np.exp(P)

def buildDictionary(lTercets):

    '''
    Create a dictionary of all words contained in the list of tercets lTercets
    The dictionary allows storing the words, and mapping each word to an index i (the corresponding index in the array of occurrencies)

    lTercets is a list of tercets (list of strings)
    '''

    hDict = {}
    nWords = 0
    for tercet in lTercets:
        words = tercet.split()
        for w in words:
            if w not in hDict:
                hDict[w] = nWords
                nWords += 1
    return hDict

def estimateModel(hlTercets, eps = 0.1):

    '''
    Build word log-probability vectors for all classes

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: tuple (h_clsLogProb, h_wordDict). h_clsLogProb is a dictionary whose keys are the classes. For each class, h_clsLogProb[cls] is an array containing, in position i, the log-frequency of the word whose index is i. h_wordDict is a dictionary that maps each word to its corresponding index.
    '''

    # Since the dictionary also includes mappings from word to indices it's more practical to build a single dict directly from the complete set of tercets, rather than doing it incrementally as we did in Solution S1
    lTercetsAll = list(itertools.chain(*hlTercets.values())) 
    hWordDict = buildDictionary(lTercetsAll)
    nWords = len(hWordDict) # Total number of words

    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = np.zeros(nWords) + eps # In this case we use 1-dimensional vectors for the model parameters. We will reshape them later.
    
    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                wordIdx = hWordDict[w]
                h_clsLogProb[cls][wordIdx] += 1 # h_clsLogProb[cls] ius a 1-D array, h_clsLogProb[cls][wordIdx] is the element in position wordIdx

    # Compute frequencies
    for cls in h_clsLogProb.keys(): # Loop over class labels
        vOccurrencies = h_clsLogProb[cls]
        vFrequencies = vOccurrencies / vOccurrencies.sum()
        vLogProbabilities = np.log(vFrequencies)
        h_clsLogProb[cls] = vLogProbabilities

    return h_clsLogProb, hWordDict
    

class Discrete_Model(Model):
    def __init__(self, n_classes, eps:float, prior: np.ndarray = -np.ones(1), label_translate= {}):
        super().__init__(n_classes,prior)
        self.eps = eps
        self.label_translate = label_translate
        

    def train(self, D, L):
        self.freqs, self.dict = estimateModel(D, self.eps)

        pass

    def validate(self, D, L):
        llm = compute_logLikelihoodMatrix(
                self.freqs,
                self.dict,
                D,
                self.label_translate,
            )
        predictions = compute_classPosteriors(
            llm,
            self.prior
        )
        acc, preds = compute_accuracy(predictions, L)
        return acc, preds, predictions

    def getConfusionMatrix(self, D, L):
        _, predL, __ = self.validate(D, L)

        
        m = np.zeros((self.n_classes, self.n_classes))
        for i in range(L.shape[0]):
            label = int(L[i])
            m[predL[i], label] += 1
        
        return m

def get_Commedia_data():
    lInf, lPur, lPar = load_data_comm()

    lInfTrain, lInfEval = split_data_comm(lInf, 4)
    lPurTrain, lPurEval = split_data_comm(lPur, 4)
    lParTrain, lParEval = split_data_comm(lPar, 4)

    hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'purgatorio': lPurTrain,
        'paradiso': lParTrain
        }


    hlTercetsTrain = {
        'inferno': lInfTrain,
        'purgatorio': lPurTrain,
        'paradiso': lParTrain
        }
    lTercetsEval = lInfEval + lPurEval + lParEval
    labelsInf = np.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = np.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = np.zeros(len(lPurEval))
    labelsPur[:] = hCls2Idx['purgatorio']


    labelsEval = np.hstack([labelsInf, labelsPur, labelsPar])


    label_translate = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    return hlTercetsTrain, lTercetsEval, labelsEval, label_translate

def get_Inf_Par():
    lInf, lPur, lPar = load_data_comm()

    lInfTrain, lInfEval = split_data_comm(lInf, 4)
    lPurTrain, lPurEval = split_data_comm(lPur, 4)
    lParTrain, lParEval = split_data_comm(lPar, 4)

    hCls2Idx = {'inferno': 0, 'paradiso': 1}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'paradiso': lParTrain
        }

    lTercetsEval = lInfEval + lParEval

    #S2_model, S2_wordDict = S2_estimateModel(hlTercetsTrain, eps = 0.001)

    labelsInf = np.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = np.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = np.hstack([labelsInf, labelsPar])

    label_translate = {'inferno': 0, 'paradiso': 1}
    return hlTercetsTrain, lTercetsEval, labelsEval, label_translate
