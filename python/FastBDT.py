#!/usr/bin/env python3

import numpy as np

import ctypes
import ctypes.util
c_double_p = ctypes.POINTER(ctypes.c_double)
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint_p = ctypes.POINTER(ctypes.c_uint)

FastBDT_library_path = ctypes.util.find_library('FastBDT_CInterface')
print('Try to load ', FastBDT_library_path)
#FastBDT_library = ctypes.cdll.LoadLibrary(FastBDT_library_path)
FastBDT_library =  ctypes.cdll.LoadLibrary('/local/ssd-scratch/tkeck/externals/development/Linux_x86_64/opt/lib64/libFastBDT_CInterface.so')
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

FastBDT_library.GetVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.GetVariableRanking.restype = ctypes.c_void_p
FastBDT_library.DeleteVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.ExtractNumberOfVariablesFromVariableRanking.argtypes = [ctypes.c_void_p]
FastBDT_library.ExtractNumberOfVariablesFromVariableRanking.restype = ctypes.c_uint
FastBDT_library.ExtractImportanceOfVariableFromVariableRanking.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT_library.ExtractImportanceOfVariableFromVariableRanking.restype = ctypes.c_double

def PrintVersion():
    FastBDT_library.PrintVersion()


class Classifier(object):
    def __init__(self, nBinningLevels=4, nTrees=100, nLayersPerTree=3, shrinkage=0.1, randRatio=0.5):
        self.forest = FastBDT_library.Create()

        FastBDT_library.SetNBinningLevels(self.forest, int(nBinningLevels))
        FastBDT_library.SetNTrees(self.forest, int(nTrees))
        FastBDT_library.SetNLayersPerTree(self.forest, int(nLayersPerTree))
        FastBDT_library.SetShrinkage(self.forest, float(shrinkage))
        FastBDT_library.SetRandRatio(self.forest, float(randRatio))

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

    def variableRanking(self):
        _ranking = FastBDT_library.GetVariableRanking(self.forest)
        ranking = []
        for i in range(FastBDT_library.ExtractNumberOfVariablesFromVariableRanking(_ranking)):
            ranking.append((i,FastBDT_library.ExtractImportanceOfVariableFromVariableRanking(_ranking, int(i))))
        FastBDT_library.DeleteVariableRanking(_ranking)
        return list(reversed(sorted(ranking, key=lambda x: x[1])))

    def __del__(self):
        FastBDT_library.Delete(self.forest)
