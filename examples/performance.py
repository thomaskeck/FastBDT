#!/usr/bin/env python3

# A python version of the performance measurement script
# I didn't used this in the paper

import sys
sys.path.append('../FastBDT/python')
sys.path.append('../xgboost/python')
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import FastBDT
import pickle
import scipy.sparse
import xgboost as xgb
import ROOT
from ROOT import TMVA
ROOT.TMVA.Tools.Instance()
import array

from timeit import default_timer as timer

class Data(object):
    def __init__(self, datafile, numberOfFeatures, numberOfEvents):
        data = np.loadtxt(datafile, skiprows=1, dtype=np.float64)
        self.numberOfFeatures = numberOfFeatures
        self.numberOfEvents = numberOfEvents
        self.X = data[:numberOfEvents, :numberOfFeatures].astype(np.float64)
        self.y = data[:numberOfEvents, -1].astype(np.uint32)


class Config(object):
    def __init__(self, numberOfFeatures, numberOfEvents, nTrees, depth, shrinkage, subSampling, nCutLevels):
        self.numberOfFeatures = numberOfFeatures
        self.numberOfEvents = numberOfEvents
        self.nTrees = nTrees
        self.depth = depth
        self.shrinkage = shrinkage
        self.subSampling = subSampling
        self.nCutLevels = nCutLevels


class Result(object):
    def __init__(self, label, probabilities, preprocessingTime, trainingTime, testTime):
        self.label = label
        self.probabilities = probabilities
        self.preprocessingTime = preprocessingTime
        self.trainingTime = trainingTime
        self.testTime = testTime


def writeResults(filename, results, test, config):
    with open(filename, 'w') as f:
        f.write("{c.nTrees} {c.depth} {c.shrinkage} {c.subSampling} {c.nCutLevels} {c.numberOfFeatures} {c.numberOfEvents}\n".format(c=config))
        f.write(" ".join(r.label for r in results) + "\n")
        f.write("PreprocessingTime: " + " ".join(str(r.preprocessingTime) for r in results) + "\n")
        f.write("TrainingTime: " + " ".join(str(r.trainingTime) for r in results) + "\n")
        f.write("TestTime: " + " ".join(str(r.testTime) for r in results) + "\n")

        for i in range(len(test.y)):
            f.write(" ".join(str(r.probabilities[i]) for r in results) + " " + str(test.y[i]) + "\n")


def measureFastBDT(train, test, config):
    preprocessing_start = timer()
    preprocessing_stop = timer()
    preprocessingTime = preprocessing_stop - preprocessing_start
    print('PreprocessingTime', preprocessingTime)
    
    training_start = timer()
    forest = FastBDT.Classifier(config.nCutLevels, config.nTrees, config.depth, config.shrinkage, config.subSampling)
    forest.fit(train.X, train.y)
    training_stop = timer()
    trainingTime = training_stop - training_start
    print('TrainingTime', trainingTime)

    test_start = timer()
    probabilities = forest.predict(test.X)
    test_stop = timer()
    testTime = test_stop - test_start
    print('TestTime', testTime)
    return Result("FastBDT", probabilities, preprocessingTime, trainingTime, testTime);


def measureSKLearn(train, test, config):
    preprocessing_start = timer()
    preprocessing_stop = timer()
    preprocessingTime = preprocessing_stop - preprocessing_start
    print('PreprocessingTime', preprocessingTime)
    
    training_start = timer()
    forest = GradientBoostingClassifier(n_estimators=config.nTrees, learning_rate=config.shrinkage, max_depth=config.depth, random_state=0, subsample=config.subSampling)
    forest.fit(train.X, train.y)
    training_stop = timer()
    trainingTime = training_stop - training_start
    print('TrainingTime', trainingTime)

    test_start = timer()
    probabilities = forest.predict_proba(test.X)[:, 1]
    test_stop = timer()
    testTime = test_stop - test_start
    print('TestTime', testTime)
    return Result("SKLearn", probabilities, preprocessingTime, trainingTime, testTime);


def measureXGBoost(train, test, config):
    preprocessing_start = timer()
    dtrain = xgb.DMatrix(train.X, label=train.y)
    dtest = xgb.DMatrix(test.X, label=test.y)
    preprocessing_stop = timer()
    preprocessingTime = preprocessing_stop - preprocessing_start
    print('PreprocessingTime', preprocessingTime)

    training_start = timer()
    param = {'max_depth':config.depth, 'eta':config.shrinkage, 'silent':1, 'objective':'binary:logistic', 'subsample': config.subSampling, 'nthread': 1}
    watchlist  = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, config.nTrees, watchlist)
    training_stop = timer()
    trainingTime = training_stop - training_start
    print('TrainingTime', trainingTime)

    test_start = timer()
    probabilities = bst.predict(dtest)
    test_stop = timer()
    testTime = test_stop - test_start
    print('TestTime', testTime)
    return Result("XGBoost", probabilities, preprocessingTime, trainingTime, testTime);


def measureTMVA(train, test, config):
    preprocessing_start = timer()
    variables = ['index', 'chiProb', 'M', 'dr', 'dz', 'E', 'p', 'pz', 'pt', 'Kid', 'piid', 'Kz', 'piz', 'Kr', 'pir', 'Kz0', 'piz0', 'pi0M',
                 'gamma1E', 'gamma2E', 'pipi0M', 'KpiM', 'Kpi0M', 'errM', 'KpCMS', 'pipCMS', 'pi0pCMS', 'distance', 'gamma1clusterTiming',
                 'gamma2clusterTiming', 'gamma1E9E25', 'gamma2E9E25', 'nTracks', 'nECLClusters', 'nKLMClusters']
    variables = variables[:config.numberOfFeatures]

    outputFile = ROOT.TFile("temp.root", "recreate")
    train_tree = ROOT.TTree("train_tree", "Training Tree")
    test_tree = ROOT.TTree("train_tree", "Training Tree")

    register = {v: array.array('f', [0]) for v in variables + ['isSignal']}
    for v in variables + ['isSignal']:
        train_tree.Branch(v, register[v], v + '/F')
        test_tree.Branch(v, register[v], v + '/F')

    for i, row in enumerate(train.X):
        for j, v in enumerate(variables):
            register[v][0] = row[j]
        register['isSignal'][0] = float(train.y[i])
        train_tree.Fill()
    
    for i, row in enumerate(test.X):
        for j, v in enumerate(variables):
            register[v][0] = row[j]
        register['isSignal'][0] = float(test.y[i])
        test_tree.Fill()
    preprocessing_stop = timer()
    preprocessingTime = preprocessing_stop - preprocessing_start
    print('PreprocessingTime', preprocessingTime)
    
    training_start = timer()
    factory = TMVA.Factory( "TMVAClassification", outputFile, 
                            "!V:Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" )
    factory.SetVerbose(False)
    for v in variables:
        factory.AddVariable(v, v, "", 'F')
    factory.SetInputTrees(train_tree, ROOT.TCut("isSignal == 1"), ROOT.TCut("isSignal == 0"))
    nsig = np.sum(train.y)
    nbkg = np.sum(1 - train.y)
    factory.PrepareTrainingAndTestTree(ROOT.TCut(""), "nTrain_Signal={}:nTrain_Background={}:SplitMode=Block:NormMode=NumEvents:!V".format(nsig, nbkg) )

    factory.BookMethod( TMVA.Types.kBDT, "BDTG",
            "!H:!V:NTrees={}:BoostType=Grad:Shrinkage={:.2f}:UseBaggedBoost:BaggedSampleFraction={:.2f}:nCuts={}:MaxDepth={}:IgnoreNegWeightsInTraining".format(config.nTrees, config.shrinkage, config.subSampling, 2**config.nCutLevels, config.depth) )                        
    factory.TrainAllMethods()
    reader = ROOT.TMVA.Reader()
    reader.SetVerbose(False)
    for v in variables:
        reader.AddVariable(v, register[v])
    reader.BookMVA("BDTG","weights/TMVAClassification_BDTG.weights.xml")
    training_stop = timer()
    trainingTime = training_stop - training_start
    print('TrainingTime', trainingTime)

    test_start = timer()
    probabilities = np.zeros(len(test.y))
    for i in range(test_tree.GetEntries()):
        test_tree.GetEvent(i)
        probabilities[i] = reader.EvaluateMVA("BDTG")
    test_stop = timer()
    testTime = test_stop - test_start
    print('TestTime', testTime)

    return Result("TMVA", probabilities, preprocessingTime, trainingTime, testTime);


i = 0
def measure(config):

    load_start = timer()
    train = Data('data/train.csv', config.numberOfFeatures, config.numberOfEvents)
    test = Data('data/test.csv', config.numberOfFeatures, config.numberOfEvents)
    load_stop = timer()
    print('Load', load_stop - load_start)

    start = timer()
    resultTMVA = measureTMVA(train, test, config)
    stop = timer()
    print('measureTMVA', stop - start)

    start = timer()
    resultFastBDT = measureFastBDT(train, test, config)
    stop = timer()
    print('measureFastBDT', stop - start)

    start = timer()
    resultSKLearn = measureSKLearn(train, test, config)
    stop = timer()
    print('measureSKLearn', stop - start)

    start = timer()
    resultXGBoost = measureXGBoost(train, test, config)
    stop = timer()
    print('measureXGBoost', stop - start)

    ++i
    writeResults('result_{}_python.txt'.format(i), [resultFastBDT, resultXGBoost, resultSKLearn, resultTMVA], test, config)


if __name__ == '__main__':

    config = Config(numberOfFeatures=35, numberOfEvents=50000, nTrees=100, shrinkage=0.1, depth=3, nCutLevels=8, subSampling=0.5)
    for i in range(35, 36):
        config.numberOfFeatures = i
        measure(config)

