#!/usr/bin/env python3

import sys

import ctypes
import ctypes.util
c_double_p = ctypes.POINTER(ctypes.c_double)
c_uint_p = ctypes.POINTER(ctypes.c_uint)

FastBDT_library_path = ctypes.util.find_library('FastBDT_CInterface')
FastBDT = ctypes.cdll.LoadLibrary(FastBDT_library_path)

FastBDT.Create.restype = ctypes.c_void_p
FastBDT.Delete.argtypes = [ctypes.c_void_p]
FastBDT.Load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
FastBDT.Train.argtypes = [ctypes.c_void_p, c_double_p, c_uint_p, ctypes.c_uint, ctypes.c_uint]
FastBDT.Analyse.argtypes = [ctypes.c_void_p, c_double_p]
FastBDT.Analyse.restype = ctypes.c_double
FastBDT.SetRandRatio.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT.SetShrinkage.argtypes = [ctypes.c_void_p, ctypes.c_double]
FastBDT.SetNTrees.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT.SetNBinningLevels.argtypes = [ctypes.c_void_p, ctypes.c_uint]
FastBDT.SetNLayersPerTree.argtypes = [ctypes.c_void_p, ctypes.c_uint]

import numpy as np


def readDataFile(datafile):
    data = np.loadtxt(datafile, skiprows=1, dtype=np.float64)
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.uint32)
    return X, y


def train():
    if len(sys.argv) < 4:
        print("Usage: ", sys.argv[0], " train datafile weightfile [nCuts=4] [nTrees=100] [nLevels=3] [shrinkage=0.1] [randRatio=0.5]")
        return 1
    
    datafile = sys.argv[2]
    weightfile = sys.argv[3]
    
    forest = FastBDT.Create()

    FastBDT.SetNBinningLevels(forest, 4)
    if len(sys.argv) > 4:
        FastBDT.SetNBinningLevels(forest, int(sys.argv[4]))

    FastBDT.SetNTrees(forest, 100)
    if len(sys.argv) > 5:
        FastBDT.SetNTrees(forest, int(sys.argv[5]))

    FastBDT.SetNLayersPerTree(forest, 3)
    if len(sys.argv) > 6:
        FastBDT.SetNLayersPerTree(forest, int(sys.argv[6]))

    FastBDT.SetShrinkage(forest, 0.1)
    if len(sys.argv) > 7:
        FastBDT.SetShrinkage(forest, float(sys.argv[7]))

    FastBDT.SetRandRatio(forest, 0.5)
    if len(sys.argv) > 8:
        FastBDT.SetRandRatio(forest, float(sys.argv[8]))

    X, y = readDataFile(datafile);

    numberOfEvents, numberOfFeatures = X.shape
    
    FastBDT.Train(forest, X.ctypes.data_as(c_double_p), y.ctypes.data_as(c_uint_p), numberOfEvents, numberOfFeatures)
    FastBDT.Save(forest, bytes(weightfile, 'utf-8'))
    analyse(forest, X, y);
    FastBDT.Delete(forest)


def analyse(forest, X, y):
    N = len(y)

    p = np.zeros(N)
    for i, row in enumerate(X):
        p[i] = FastBDT.Analyse(forest, row.ctypes.data_as(c_double_p))
    
    print("Fraction of correctly categorised samples {:.4f}".format( np.sum((p > 0.5) & (y == 1) | (p <= 0.5) & (y != 1)) / N ))
    print("Signal Efficiency {:.4f}".format( np.sum((p > 0.5) & (y == 1)) / np.sum(y == 1) ))
    print("Background Efficiency {:.4f}".format( np.sum((p <= 0.5) & (y != 1)) / np.sum(y != 1) ))
    print("Signal Purity {:.4f}".format( np.sum((p > 0.5) & (y == 1)) / np.sum(p > 0.5) ))


def apply():
    if len(sys.argv) < 4:
        print("Usage: ", sys.argv[0], " apply datafile weightfile")
        return 1

    datafile = sys.argv[2]
    weightfile = sys.argv[3]

    X, y = readDataFile(datafile)

    forest = FastBDT.Create()
    FastBDT.Load(forest, bytes(weightfile, 'utf-8'))
    analyse(forest, X, y);
    FastBDT.Delete(forest)
    return 0


def output():
    if len(sys.argv) < 4:
        print("Usage: ", sys.argv[0], " output datafile weightfile")
        return 1

    datafile = sys.argv[2]
    weightfile = sys.argv[3]

    X, y = readDataFile(datafile)

    forest = FastBDT.Create()
    FastBDT.Load(forest, bytes(weightfile, 'utf-8'))
    for i, row in enumerate(X):
        print(int(y[i] == 1), FastBDT.Analyse(forest, row.ctypes.data_as(c_double_p)))
    FastBDT.Delete(forest)
    return 0


if __name__ == '__main__':

    FastBDT.PrintVersion()

    if len(sys.argv) <= 1:
        print("Usage ", sys.argv[0], " [train|apply|output]")
        sys.exit(1)

    if sys.argv[1] == 'train':
        ret = train()
    elif sys.argv[1] == 'apply':
        ret = apply()
    elif sys.argv[1] == 'output':
        ret = output()
    else:
        print("Unkown option", sys.argv[1]) 
        ret = 1

    sys.exit(ret)

