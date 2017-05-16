#!/usr/bin/env python3

import os
import numpy as np

import ctypes
import ctypes.util
c_double_p = ctypes.POINTER(ctypes.c_double)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_bool_p = ctypes.POINTER(ctypes.c_bool)
c_uint_p = ctypes.POINTER(ctypes.c_uint)

FastBDT_library =  ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libFastBDT_CInterface.so'))

FastBDT_library.Create.restype = ctypes.c_void_p
FastBDT_library.Delete.argtypes = [ctypes.c_void_p]

FastBDT_library.Load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
FastBDT_library.Save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

FastBDT_library.Fit.argtypes = [ctypes.c_void_p, c_float_p, c_float_p, c_bool_p, ctypes.c_uint]

FastBDT_library.Predict.argtypes = [ctypes.c_void_p, c_float_p]
FastBDT_library.Predict.restype = ctypes.c_float

FastBDT_library.PredictArray.argtypes = [ctypes.c_void_p, c_float_p, c_float_p, ctypes.c_uint]

FastBDT_library.SetSubsample.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT_library.GetSubsample.argtypes = [ctypes.c_void_p]
FastBDT_library.GetSubsample.restypes = ctypes.c_double

FastBDT_library.SetShrinkage.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT_library.GetShrinkage.argtypes = [ctypes.c_void_p]
FastBDT_library.GetShrinkage.restypes = ctypes.c_double

FastBDT_library.SetFlatnessLoss.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT_library.GetFlatnessLoss.argtypes = [ctypes.c_void_p]
FastBDT_library.GetFlatnessLoss.restypes = ctypes.c_double

FastBDT_library.SetNTrees.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.GetNTrees.argtypes = [ctypes.c_void_p]
FastBDT_library.GetNTrees.restypes = ctypes.c_uint

FastBDT_library.SetBinning.argtypes = [ctypes.c_void_p, c_uint_p, ctypes.c_uint]
FastBDT_library.SetPurityTransformation.argtypes = [ctypes.c_void_p, c_uint_p, ctypes.c_uint]

FastBDT_library.SetDepth.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.GetDepth.argtypes = [ctypes.c_void_p]
FastBDT_library.GetDepth.restypes = ctypes.c_uint

FastBDT_library.SetTransform2Probability.argtypes = [ctypes.c_void_p, ctypes.c_bool]
FastBDT_library.GetTransform2Probability.argtypes = [ctypes.c_void_p]
FastBDT_library.GetTransform2Probability.restypes = ctypes.c_bool

FastBDT_library.SetSPlot.argtypes = [ctypes.c_void_p, ctypes.c_bool]
FastBDT_library.GetSPlot.argtypes = [ctypes.c_void_p]
FastBDT_library.GetSPlot.restypes = ctypes.c_bool


FastBDT_library.GetVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.GetVariableRanking.restype = ctypes.c_void_p
FastBDT_library.DeleteVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.ExtractNumberOfVariablesFromVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.ExtractNumberOfVariablesFromVariableRanking.restype = ctypes.c_uint
FastBDT_library.ExtractImportanceOfVariableFromVariableRanking.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.ExtractImportanceOfVariableFromVariableRanking.restype = ctypes.c_double
    
FastBDT_library.GetIndividualVariableRanking.argtypes = [ctypes.c_void_p, c_float_p]
FastBDT_library.GetIndividualVariableRanking.restype = ctypes.c_void_p


def PrintVersion():
    FastBDT_library.PrintVersion()


def calculate_roc_auc(p, t, w=None):
    """
    Calculates the area under the receiver oeprating characteristic curve (AUC ROC)
    @param p np.array filled with the probability output of a classifier
    @param t np.array filled with the target (0 or 1)
    """
    if w is None:
        w = np.ones(len(t))
    N = w.sum()
    T = np.sum(t*w)
    t = t*w
    index = np.argsort(p)
    efficiency = (T - np.cumsum(t[index])) / float(T)
    purity = (T - np.cumsum(t[index])) / (N - np.cumsum(w))
    purity = np.where(np.isnan(purity), 0, purity)
    return np.abs(np.trapz(purity, efficiency))


class Classifier(object):
    def __init__(self, binning=[], nTrees=100, depth=3, shrinkage=0.1, subsample=0.5, transform2probability=True, purityTransformation=[], sPlot=False, flatnessLoss=-1.0, numberOfFlatnessFeatures=0):
        """
        @param binning list of numbers with the power N used for each feature binning e.g. 8 means 2^8 bins
        @param nTrees number of trees
        @param shrinkage reduction factor of each tree, lower shrinkage leads to slower but more stable convergence
        @param subsample the ratio of samples used for each tree
        @param transform2probability whether to transform the output to a probability
        @param purityTransformation list of bools, defines for each feature of in addition the purity-transformed of the feature should be used (this will slow down the inference)
        @param sPlot special treatment of sPlot weights are used
        @param flatnessLoss if bigger than 0 a flatness boost against all flatnessFeatures
        @param numberOfFlatnessFeatures the number of flatness features, it is assumed that the last N features are the flatness features
        """
        self.binning = binning
        self.nTrees = nTrees
        self.depth = depth
        self.shrinkage = shrinkage
        self.subsample = subsample
        self.transform2probability = transform2probability
        self.purityTransformation = purityTransformation
        self.sPlot = sPlot
        self.flatnessLoss = flatnessLoss
        self.numberOfFlatnessFeatures = numberOfFlatnessFeatures
        self.forest = self.create_forest()

    def create_forest(self):
        forest = FastBDT_library.Create()
        FastBDT_library.SetBinning(forest, np.array(self.binning).ctypes.data_as(c_uint_p), int(len(self.binning)))
        FastBDT_library.SetNTrees(forest, int(self.nTrees))
        FastBDT_library.SetDepth(forest, int(self.depth))
        FastBDT_library.SetNumberOfFlatnessFeatures(forest, int(self.numberOfFlatnessFeatures))
        FastBDT_library.SetShrinkage(forest, float(self.shrinkage))
        FastBDT_library.SetSubsample(forest, float(self.subsample))
        FastBDT_library.SetFlatnessLoss(forest, float(self.flatnessLoss))
        FastBDT_library.SetTransform2Probability(forest, bool(self.transform2probability))
        FastBDT_library.SetSPlot(forest, bool(self.sPlot))
        FastBDT_library.SetPurityTransformation(forest, np.array(self.purityTransformation).ctypes.data_as(c_uint_p), int(len(self.purityTransformation)))
        return forest

    def fit(self, X, y, weights=None):
        X_temp = np.require(X, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
        y_temp = np.require(y, dtype=np.bool, requirements=['A', 'W', 'C', 'O'])
        if weights is not None:
            w_temp = np.require(weights, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
        numberOfEvents, numberOfFeatures = X_temp.shape
        FastBDT_library.Fit(self.forest, X_temp.ctypes.data_as(c_float_p),
                              w_temp.ctypes.data_as(c_float_p) if weights is not None else None,
                              y_temp.ctypes.data_as(c_bool_p), int(numberOfEvents), int(numberOfFeatures))
        return self

    def predict(self, X):
        X_temp = np.require(X, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
        N = len(X)
        p = np.require(np.zeros(N), dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
        FastBDT_library.PredictArray(self.forest, X_temp.ctypes.data_as(c_float_p), p.ctypes.data_as(c_float_p), int(X_temp.shape[0]))
        return p
    
    def predict_single(self, row):
        return FastBDT_library.Predict(self.forest, row.ctypes.data_as(c_float_p))

    def save(self, weightfile):
        FastBDT_library.Save(self.forest, bytes(weightfile, 'utf-8'))

    def load(self, weightfile):
        FastBDT_library.Load(self.forest, bytes(weightfile, 'utf-8'))
    
    def individualFeatureImportance(self, X):
        X_temp = np.require(X, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
        _ranking = FastBDT_library.GetIndividualVariableRanking(self.forest, X_temp.ctypes.data_as(c_float_p))
        ranking = dict()
        for i in range(FastBDT_library.ExtractNumberOfVariablesFromVariableRanking(_ranking)):
            ranking[i] = FastBDT_library.ExtractImportanceOfVariableFromVariableRanking(_ranking, int(i))
        FastBDT_library.DeleteVariableRanking(_ranking)
        return ranking

    def internFeatureImportance(self):
        _ranking = FastBDT_library.GetVariableRanking(self.forest)
        ranking = dict()
        for i in range(FastBDT_library.ExtractNumberOfVariablesFromVariableRanking(_ranking)):
            ranking[i] = FastBDT_library.ExtractImportanceOfVariableFromVariableRanking(_ranking, int(i))
        FastBDT_library.DeleteVariableRanking(_ranking)
        return ranking

    def externFeatureImportance(self, X, y, weights=None, X_test=None, y_test=None, weights_test=None):
        if X_test is None:
            X_test = X
        if y_test is None:
            y_test = y
        if weights_test is None:
            weights_test = weights
        numberOfEvents, numberOfFeatures = X.shape
        global_auc = calculate_roc_auc(self.predict(X_test), y_test, weights_test)
        forest = self.forest
        importances = self._externFeatureImportance(list(range(numberOfFeatures)), global_auc, X, y, weights, X_test, y_test, weights_test)
        self.forest = forest
        return importances

    def _externFeatureImportance(self, features, global_auc, X, y, weights, X_test, y_test, weights_test):
        importances = dict()
        for i in features:
            remaining_features = [f for f in features if f != i]
            X_temp = X[:, remaining_features]
            X_test_temp = X_test[:, remaining_features]
            self.forest = self.create_forest()
            self.fit(X_temp, y, weights)
            auc = calculate_roc_auc(self.predict(X_test_temp), y_test, weights_test)
            FastBDT_library.Delete(self.forest)
            importances[i] = global_auc - auc
   
        most_important = max(importances.keys(), key=lambda x: importances[x])
        remaining_features = [v for v in features if v != most_important]
        if len(remaining_features) == 1:
            return importances
    
        importances = {most_important: importances[most_important]}
        rest = self._externFeatureImportance(remaining_features, global_auc - importances[most_important], X, y, weights, X_test, y_test, weights_test)
        importances.update(rest)
        return importances

    def __del__(self):
        FastBDT_library.Delete(self.forest)
