#!/usr/bin/env python3

import sys

import numpy as np

from PyFastBDT import FastBDT

def readDataFile(datafile):
    data = np.loadtxt(datafile, skiprows=1, dtype=np.float64)
    X = data[:, :-1].astype(np.float64)
    #w = data[:, -2].astype(np.float64)
    w = None
    y = data[:, -1].astype(np.uint32)
    return X, y, w


def train():
    if len(sys.argv) < 4:
        print("Usage: ", sys.argv[0], " train datafile weightfile [nCuts=4] [nTrees=100] [nLevels=3] [shrinkage=0.1] [randRatio=0.5]")
        return 1
    
    datafile = sys.argv[2]
    weightfile = sys.argv[3]
    
    forest = FastBDT.Classifier(*sys.argv[4:])
    X, y, w = readDataFile(datafile);

    forest.fit(X, y, w)
    forest.save(weightfile)
    analyse(forest, X, y);


def analyse(forest, X, y):
    N = len(y)
    p = forest.predict(X)
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

    X, y, w = readDataFile(datafile)

    forest = FastBDT.Classifier()
    forest.load(weightfile)
    analyse(forest, X, y);
    return 0


def output():
    if len(sys.argv) < 4:
        print("Usage: ", sys.argv[0], " output datafile weightfile")
        return 1

    datafile = sys.argv[2]
    weightfile = sys.argv[3]

    X, y, w = readDataFile(datafile)

    forest = FastBDT.Classifier()
    forest.load(weightfile)
    
    p = forest.predict(X)
    for i, p in enumerate(p):
        print(int(y[i] == 1), p)
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

