#!/usr/bin/env python3

import numpy as np

import ctypes
import ctypes.util
c_double_p = ctypes.POINTER(ctypes.c_double)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint_p = ctypes.POINTER(ctypes.c_uint)

import os

# Apparently find_library does not work as I expected
#FastBDT_library_path = ctypes.util.find_library('FastBDT_CInterface')
#print('Try to load ', FastBDT_library_path)
#FastBDT_library = ctypes.cdll.LoadLibrary(FastBDT_library_path)

FastBDT_library =  ctypes.cdll.LoadLibrary(os.getcwd() + '/libFastBDT_CInterface.so')
print('Loaded ', FastBDT_library)

FastBDT_library.Create.restype = ctypes.c_void_p
FastBDT_library.Delete.argtypes = [ctypes.c_void_p]
FastBDT_library.Load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
FastBDT_library.Train.argtypes = [ctypes.c_void_p, c_double_p, c_float_p, c_uint_p, ctypes.c_uint, ctypes.c_uint]
FastBDT_library.Analyse.argtypes = [ctypes.c_void_p, c_double_p]
FastBDT_library.Analyse.restype = ctypes.c_double
FastBDT_library.AnalyseArray.argtypes = [ctypes.c_void_p, c_double_p, c_double_p, ctypes.c_uint, ctypes.c_uint]
FastBDT_library.SetRandRatio.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT_library.SetShrinkage.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT_library.SetNTrees.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.SetNBinningLevels.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.SetNLayersPerTree.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.SetTransform2Probability.argtypes = [ctypes.c_void_p, ctypes.c_bool]
FastBDT_library.SetPurityTransformation.argtypes = [ctypes.c_void_p, ctypes.c_uint]

FastBDT_library.GetVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.GetVariableRanking.restype = ctypes.c_void_p
FastBDT_library.DeleteVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.ExtractNumberOfVariablesFromVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.ExtractNumberOfVariablesFromVariableRanking.restype = ctypes.c_uint
FastBDT_library.ExtractImportanceOfVariableFromVariableRanking.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.ExtractImportanceOfVariableFromVariableRanking.restype = ctypes.c_double


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
    def __init__(self, nBinningLevels=4, nTrees=100, nLayersPerTree=3, shrinkage=0.1, randRatio=0.5, transform2probability=True, purityTransformation=0):
        self.nBinningLevels = nBinningLevels
        self.nTrees = nTrees
        self.nLayersPerTree = nLayersPerTree
        self.shrinkage = shrinkage
        self.randRatio = randRatio
        self.transform2probability = transform2probability
        self.purityTransformation = purityTransformation
        self.forest = self.create_forest()

    def create_forest(self):
        forest = FastBDT_library.Create()
        FastBDT_library.SetNBinningLevels(forest, int(self.nBinningLevels))
        FastBDT_library.SetNTrees(forest, int(self.nTrees))
        FastBDT_library.SetNLayersPerTree(forest, int(self.nLayersPerTree))
        FastBDT_library.SetShrinkage(forest, float(self.shrinkage))
        FastBDT_library.SetRandRatio(forest, float(self.randRatio))
        FastBDT_library.SetTransform2Probability(forest, bool(self.transform2probability))
        FastBDT_library.SetPurityTransformation(forest, int(self.purityTransformation))
        return forest

    def fit(self, X, y, weights=None):
        X_temp = np.require(X, dtype=np.float64, requirements=['A', 'W', 'C', 'O'])
        y_temp = np.require(y, dtype=np.uint32, requirements=['A', 'W', 'C', 'O'])
        if weights is not None:
            w_temp = np.require(weights, dtype=np.float32, requirements=['A', 'W', 'C', 'O'])
        numberOfEvents, numberOfFeatures = X_temp.shape
        FastBDT_library.Train(self.forest, X_temp.ctypes.data_as(c_double_p),
                              w_temp.ctypes.data_as(c_float_p) if weights is not None else None,
                              y_temp.ctypes.data_as(c_uint_p), int(numberOfEvents), int(numberOfFeatures))
        return self

    def predict(self, X):
        X_temp = np.require(X, dtype=np.float64, requirements=['A', 'W', 'C', 'O'])
        N = len(X)
        p = np.require(np.zeros(N), dtype=np.float64, requirements=['A', 'W', 'C', 'O'])
        FastBDT_library.AnalyseArray(self.forest, X_temp.ctypes.data_as(c_double_p), p.ctypes.data_as(c_double_p), int(X_temp.shape[0]), int(X_temp.shape[1]))
        return p
    
    def predict_single(self, row):
        return FastBDT_library.Analyse(self.forest, row.ctypes.data_as(c_double_p))

    def save(self, weightfile):
        FastBDT_library.Save(self.forest, bytes(weightfile, 'utf-8'))

    def load(self, weightfile):
        FastBDT_library.Load(self.forest, bytes(weightfile, 'utf-8'))
    
    def individualFeatureImportance(self, X):
        X_temp = np.require(X, dtype=np.float64, requirements=['A', 'W', 'C', 'O'])
        _ranking = FastBDT_library.GetIndividualVariableRanking(self.forest, X_temp.ctypes.data_as(c_double_p))
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
        self.forest = self.create_forest()
        importances = self._externFeatureImportance(list(range(numberOfFeatures)), global_auc, X, y, weights, X_test, y_test, weights_test)
        FastBDT_library.Delete(self.forest)
        self.forest = forest
        return importances

    def _externFeatureImportance(self, features, global_auc, X, y, weights, X_test, y_test, weights_test):
        importances = dict()
        for i in features:
            remaining_features = [f for f in features if f != i]
            X_temp = X[:, remaining_features]
            X_test_temp = X_test[:, remaining_features]
            self.fit(X_temp, y, weights)
            auc = calculate_roc_auc(self.predict(X_test_temp), y_test, weights_test)
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
