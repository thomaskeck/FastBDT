/**
 * Thomas Keck 2017
 *
 * Performance comparison code.
 *
 * This file measures the runtime of the fitting and application phase for different hyper-parameter settings for
 *  - FastBDT
 *  - XGBoost
 *  - TMVA
 *  - SKLearn
 * All methods are accessed via C++ to ensure an optimal performance.
 * In case of XGBoost and SKLearn this is rather unusual, but it is faster than to use Python.
 * I wouldn't recommend using XGBoost and SKLearn in the way it is shown below in your daily work,
 * it is rather error-prone, and I only do this to ensure fair conditions between the contestants.
 *
 * Compiling this code is complicated because it involves all the different frameworks.
 * You have to install FastBDT and XGBoost from github, ROOT, and sklearn using pip3; as well as python3.5 headers and libraries for your distribution.
 *
 * I compile in the following way
 * g++ comparison.cxx -o comparison -O3
 *   -L ../FastBDT/ -I ../FastBDT/include/ -lFastBDT_shared
 *   -I ../xgboost/rabit/include/ -I ../xgboost/dmlc-core/include/ -L ../xgboost/rabit/lib/ -L ../xgboost/dmlc-core/ -I ../xgboost/include/ -L ../xgboost/lib/ -l xgboost
 *   `root-config --cflags --libs` -lTMVA -lMLP -lXMLIO
 *   -lpython3.5
 *
 * And execute it like this:
 * LD_LIBRARY_PATH=$PATH_TO_XGBOOST/lib/:$PATH_TO_FASTBDT/:$LD_LIBRARY_PATH ./comparison 10
 *
 * The executable takes a command line argument, which I use to call the executable multiple times
 * with different hyper-parameter configurations.
 *
 * The code reads its data from data/train.csv and data/test.csv
 * These files must contain the appropriate amount of features (separated by whitespaces) and events (separated by linebreaks).
 * The last feature must be an integer with the truth information (1 for signal and 0 for background).
 *
 * The results of the measurements are outputted in files result_$id_cpp.txt, which contain the runtime in the preprocessing (preparation of the data),
 * fitting (fitting the classifier) and application (inference on indepentend test data using the classifier); as well as the outputted probabilities
 * for the test dataset for each event (where the last column contains the truth variable).
 */

#include "FastBDT.h"

#include "xgboost/c_api.h"
#include "xgboost/data.h"

#include <TMVA/Factory.h>
#include <TMVA/Reader.h>
#include <TMVA/Tools.h>
#include <TMVA/DataLoader.h>

#include <TFile.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TString.h>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>

class Data {
  public:
    Data(std::string datafile, unsigned int _numberOfFeatures, unsigned int _numberOfEvents) : numberOfFeatures(_numberOfFeatures), numberOfEvents(_numberOfEvents) {

      X.reserve(numberOfEvents);
      y.reserve(numberOfEvents);

      std::fstream fs (datafile, std::fstream::in);
      std::string line;

      // Skip Header
      std::getline(fs, line);

      unsigned int iEvent = 0;
      while(std::getline(fs, line)) {

        std::istringstream sin(line);
        std::vector<float> row;
        float value = 0;
        unsigned int iFeature = 0;
        while(sin >> value) {
          if(iFeature < numberOfFeatures)
            row.push_back(value);
          ++iFeature;
        }
        X.push_back(row);
        y.push_back(static_cast<int>(value));

        ++iEvent;
        if(iEvent >= numberOfEvents) {
          break;
        }
      }

      std::cout << "Loaded " << iEvent << " Events" << std::endl;
    }

    unsigned int numberOfEvents = 0;
    unsigned int numberOfFeatures = 0;
    std::vector<std::vector<float>> X;
    std::vector<unsigned int> y;
};


struct Config {
    unsigned int nTrees;
    unsigned int depth;
    double shrinkage;
    double subSampling;
    // Only TMVA and FastBDT
    unsigned int nCutLevels;
    unsigned int numberOfFeatures;
    unsigned int numberOfEvents;
};

struct Result {

  std::string label;
  std::vector<double> probabilities;
  std::chrono::duration<double, std::milli> preprocessingTime;
  std::chrono::duration<double, std::milli> trainingTime;
  std::chrono::duration<double, std::milli> testTime;
};


void writeResults(std::string filename, const std::vector<Result> &results, const Data& test, const Config& config) {

  std::fstream str(filename, std::fstream::out);
  str << config.nTrees << " " << config.depth << " " << config.shrinkage << " " << config.subSampling << " " << config.nCutLevels << " " << config.numberOfFeatures << " " << config.numberOfEvents << std::endl;

  str << "Labels: ";
  for(auto &r : results) {
    str << r.label << " ";
  }
  str << std::endl;
  
  str << "PreprocessingTime: ";
  for(auto &r : results) {
    str << r.preprocessingTime.count() << " ";
  }
  str << std::endl;
  
  str << "TrainingTime: ";
  for(auto &r : results) {
    str << r.trainingTime.count() << " ";
  }
  str << std::endl;
  
  str << "TestTime: ";
  for(auto &r : results) {
    str << r.testTime.count() << " ";
  }
  str << std::endl;

  for(unsigned int iEvent = 0; iEvent < config.numberOfEvents;  ++iEvent) {
    for(auto &r : results) {
        str << r.probabilities[iEvent] << " ";
    }
    str << test.y[iEvent] << std::endl;
  }

}

Result measureSKLearn(const Data& train, const Data& test, const Config& config) {
    
    Result result;
    result.label = "SKLearn";
    
    std::chrono::high_resolution_clock::time_point preprocessingTime1 = std::chrono::high_resolution_clock::now();
    PyObject* cls = PyUnicode_FromString((char*)"GradientBoostingClassifier");
    PyObject* fit = PyUnicode_FromString((char*)"fit");
    PyObject* predict = PyUnicode_FromString((char*)"predict_proba");
    PyObject* pModule = PyImport_ImportModule("sklearn.ensemble");
    
    PyObject* loss = PyUnicode_FromString((char*)"deviance");
    PyObject* learning_rate = PyFloat_FromDouble(static_cast<double>(config.shrinkage));
    PyObject* n_estimators = PyLong_FromLong(static_cast<long>(config.nTrees));
    PyObject* subsample = PyFloat_FromDouble(static_cast<double>(config.subSampling));
    PyObject* criterion = PyUnicode_FromString((char*)"friedman_mse");
    PyObject* min_samples_split = PyLong_FromLong(static_cast<long>(2));
    PyObject* min_samples_leaf = PyLong_FromLong(static_cast<long>(1));
    PyObject* min_weight_fraction_leaf = PyFloat_FromDouble(static_cast<double>(0.0));
    PyObject* max_depth = PyLong_FromLong(static_cast<long>(config.depth));
    PyObject* forest = PyObject_CallMethodObjArgs(pModule, cls, loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, NULL);
    Py_DECREF(loss);
    Py_DECREF(learning_rate);
    Py_DECREF(n_estimators);
    Py_DECREF(subsample);
    Py_DECREF(min_samples_split);
    Py_DECREF(min_samples_leaf);
    Py_DECREF(min_weight_fraction_leaf);
    Py_DECREF(max_depth);
    
    float *X = new float[train.numberOfEvents*train.numberOfFeatures];
    float *y = new float[train.numberOfEvents];
    for(unsigned int iEvent = 0; iEvent < train.numberOfEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature)
            X[iEvent*train.numberOfFeatures + iFeature] = train.X[iEvent][iFeature];
        y[iEvent] = static_cast<float>(train.y[iEvent]);
    }
    long dimensions_X[2] = {train.numberOfEvents, train.numberOfFeatures};
    long dimensions_y[1] = {train.numberOfEvents};
    PyObject* ndarray_X = PyArray_SimpleNewFromData(2, dimensions_X, NPY_FLOAT32, X);
    PyObject* ndarray_y = PyArray_SimpleNewFromData(1, dimensions_y, NPY_FLOAT32, y);

    std::chrono::high_resolution_clock::time_point preprocessingTime2 = std::chrono::high_resolution_clock::now();
    result.preprocessingTime = preprocessingTime2 - preprocessingTime1;
    std::cout << "PreprocessingTime " << result.preprocessingTime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point trainingTime1 = std::chrono::high_resolution_clock::now();
    PyObject *x = PyObject_CallMethodObjArgs(forest, fit, ndarray_X, ndarray_y, NULL);
    std::chrono::high_resolution_clock::time_point trainingTime2 = std::chrono::high_resolution_clock::now();
    result.trainingTime = trainingTime2 - trainingTime1;
    std::cout << "TrainingTime " << result.trainingTime.count() << std::endl;

    result.probabilities.resize(test.numberOfEvents);
    
    std::chrono::high_resolution_clock::time_point testTime1 = std::chrono::high_resolution_clock::now();
    float *X_test = new float[test.numberOfEvents*test.numberOfFeatures];
    for(unsigned int iEvent = 0; iEvent < test.numberOfEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < test.numberOfFeatures; ++iFeature)
            X_test[iEvent*test.numberOfFeatures + iFeature] = test.X[iEvent][iFeature];
    }
    long dimensions_X_test[2] = {test.numberOfEvents, test.numberOfFeatures};
    PyObject* ndarray_X_test = PyArray_SimpleNewFromData(2, dimensions_X_test, NPY_FLOAT32, X_test);

    PyObject *pyresult = PyObject_CallMethodObjArgs(forest, predict, ndarray_X_test, NULL);
    for(unsigned int iEvent = 0; iEvent < test.numberOfEvents; ++iEvent) {
        result.probabilities[iEvent] = 1.0 - static_cast<float>(*static_cast<double*>(PyArray_GETPTR1(pyresult, iEvent)));
    }
    std::chrono::high_resolution_clock::time_point testTime2 = std::chrono::high_resolution_clock::now();
    result.testTime = testTime2 - testTime1;
    std::cout << "TestTime " << result.testTime.count() << std::endl;
    
    Py_DECREF(ndarray_X);
    Py_DECREF(ndarray_X_test);
    Py_DECREF(ndarray_y);
    Py_DECREF(pyresult);
    Py_DECREF(cls);
    Py_DECREF(predict);
    Py_DECREF(fit);
    Py_DECREF(pModule);
    delete[] X;
    delete[] X_test;
    delete[] y;

    return result;

}

Result measureFastBDT(const Data& train, const Data& test, const Config& config) {
    
    Result result;
    result.label = "FastBDT";

    std::chrono::high_resolution_clock::time_point preprocessingTime1 = std::chrono::high_resolution_clock::now();
    // Equal statistics binning
    std::vector<FastBDT::FeatureBinning<float>> featureBinnings(train.numberOfFeatures);
    std::vector<float> feature(train.numberOfEvents);
    for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature) {
        for(unsigned int iEvent = 0; iEvent < train.numberOfEvents; ++iEvent)
          feature[iEvent] = train.X[iEvent][iFeature];
        featureBinnings[iFeature] = FastBDT::FeatureBinning<float>(config.nCutLevels, feature);
    }

    // Fill event Sample
    FastBDT::EventSample eventSample(train.numberOfEvents, train.numberOfFeatures, 0, std::vector<unsigned int>(train.numberOfFeatures, config.nCutLevels));
    std::vector<unsigned int> bins(train.numberOfFeatures);
    for(unsigned int iEvent = 0; iEvent < train.numberOfEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature)
            bins[iFeature] = featureBinnings[iFeature].ValueToBin( train.X[iEvent][iFeature] );
        eventSample.AddEvent(bins, 1.0, train.y[iEvent] == 1);
    }
    std::chrono::high_resolution_clock::time_point preprocessingTime2 = std::chrono::high_resolution_clock::now();
    result.preprocessingTime = preprocessingTime2 - preprocessingTime1;
    std::cout << "PreprocessingTime " << result.preprocessingTime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point trainingTime1 = std::chrono::high_resolution_clock::now();
    // Train classifier using training data
    FastBDT::ForestBuilder dt(eventSample, config.nTrees, config.shrinkage, config.subSampling, config.depth);
    FastBDT::Forest<float> forest( dt.GetShrinkage(), dt.GetF0(), false);
    for( auto t : dt.GetForest() )
        forest.AddTree(FastBDT::removeFeatureBinningTransformationFromTree(t, featureBinnings));
    std::chrono::high_resolution_clock::time_point trainingTime2 = std::chrono::high_resolution_clock::now();
    result.trainingTime = trainingTime2 - trainingTime1;
    std::cout << "TrainingTime " << result.trainingTime.count() << std::endl;

    result.probabilities.resize(test.numberOfEvents);
    
    // Apply classifier on test data
    std::chrono::high_resolution_clock::time_point testTime1 = std::chrono::high_resolution_clock::now();
    for(unsigned int iEvent = 0; iEvent < test.numberOfEvents; ++iEvent) {
      result.probabilities[iEvent] = forest.Analyse(test.X[iEvent]);
    }
    std::chrono::high_resolution_clock::time_point testTime2 = std::chrono::high_resolution_clock::now();
    result.testTime = testTime2 - testTime1;
    std::cout << "TestTime " << result.testTime.count() << std::endl;
    return result;

}

Result measureTMVA(const Data& train, const Data& test, const Config& config) {
    
    Result result;
    result.label = "TMVA";

    std::chrono::high_resolution_clock::time_point preprocessingTime1 = std::chrono::high_resolution_clock::now();
    TMVA::Tools::Instance();
    TFile classFile("TMVA.root", "RECREATE");
    classFile.cd();
    TMVA::DataLoader data_loader("TMVAClassification");
    TMVA::Factory factory("TMVAClassification", &classFile, "!V:Silent:Color:DrawProgressBar:AnalysisType=Classification");

    std::vector<std::string> variables = {"M", "p", "pt", "pz", "phi", "daughter__bo0__cm__spp__bc",       "daughter__bo0__cm__sppz__bc", "daughter__bo0__cm__sppt__bc",       "daughter__bo0__cm__spphi__bc", "daughter__bo1__cm__spp__bc",       "daughter__bo1__cm__sppz__bc", "daughter__bo1__cm__sppt__bc",       "daughter__bo1__cm__spphi__bc", "daughter__bo2__cm__spp__bc",       "daughter__bo2__cm__sppz__bc", "daughter__bo2__cm__sppt__bc",       "daughter__bo2__cm__spphi__bc", "chiProb", "dr", "dz", "dphi",       "daughter__bo0__cm__spdr__bc", "daughter__bo1__cm__spdr__bc",       "daughter__bo0__cm__spdz__bc", "daughter__bo1__cm__spdz__bc",       "daughter__bo0__cm__spdphi__bc", "daughter__bo1__cm__spdphi__bc",       "daughter__bo0__cm__spchiProb__bc", "daughter__bo1__cm__spchiProb__bc",       "daughter__bo2__cm__spchiProb__bc", "daughter__bo0__cm__spKid__bc",       "daughter__bo0__cm__sppiid__bc", "daughter__bo1__cm__spKid__bc",       "daughter__bo1__cm__sppiid__bc", "daughterAngle__bo0__cm__sp1__bc",       "daughterAngle__bo0__cm__sp2__bc", "daughterAngle__bo1__cm__sp2__bc",       "daughter__bo2__cm__spdaughter__bo0__cm__spE__bc__bc",       "daughter__bo2__cm__spdaughter__bo1__cm__spE__bc__bc",       "daughter__bo2__cm__spdaughter__bo0__cm__spclusterTiming__bc__bc",       "daughter__bo2__cm__spdaughter__bo1__cm__spclusterTiming__bc__bc",       "daughter__bo2__cm__spdaughter__bo0__cm__spclusterE9E25__bc__bc",       "daughter__bo2__cm__spdaughter__bo1__cm__spclusterE9E25__bc__bc",       "daughter__bo2__cm__spdaughter__bo0__cm__spminC2HDist__bc__bc",       "daughter__bo2__cm__spdaughter__bo1__cm__spminC2HDist__bc__bc",       "daughterInvariantMass__bo0__cm__sp1__bc",       "daughterInvariantMass__bo0__cm__sp2__bc",       "daughterInvariantMass__bo1__cm__sp2__bc"};

    std::vector<float> vec(train.numberOfFeatures);
    for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature) {
      data_loader.AddVariable(variables[iFeature].c_str());
    }

    TTree *signal_tree = new TTree("signal_tree", "signal_tree");
    TTree *background_tree = new TTree("background_tree", "background_tree");

    for (unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature) {
      signal_tree->Branch(variables[iFeature].c_str(), &vec[iFeature]);
      background_tree->Branch(variables[iFeature].c_str(), &vec[iFeature]);
    }

    unsigned int nsig = 0;
    unsigned int nbkg = 0;
    for(unsigned int iEvent = 0; iEvent < train.numberOfEvents; ++iEvent) {
      for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature) {
        vec[iFeature] = train.X[iEvent][iFeature];
      }
      if(train.y[iEvent] == 1) {
        ++nsig;
        signal_tree->Fill();
      } else {
        ++nbkg;
        background_tree->Fill();
      }
    }

    data_loader.AddSignalTree(signal_tree);
    data_loader.AddBackgroundTree(background_tree);

    data_loader.PrepareTrainingAndTestTree("", std::string("nTrain_Signal=") + std::to_string(nsig) + std::string(":nTrain_Background=") + std::to_string(nbkg) + std::string(":SplitMode=Block:!V"));
    factory.BookMethod(&data_loader, TMVA::Types::kBDT, "BDTG", std::string("!H:!V:NTrees=") + std::to_string(config.nTrees) + std::string("BoostType=Grad:Shrinkage=") + std::to_string(config.shrinkage) + std::string(":UseBaggedBoost:BaggedSampleFraction=") + std::to_string(config.subSampling) + std::string(":nCuts=") + std::to_string(1 << config.nCutLevels) + std::string(":MaxDepth=") + std::to_string(config.depth) + std::string(":IgnoreNegWeightsInTraining"));

    TMVA::Reader *reader = new TMVA::Reader("!Color:!Silent");
    for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature) {
      reader->AddVariable(variables[iFeature].c_str(), &vec[iFeature]);
    }
    
    std::chrono::high_resolution_clock::time_point preprocessingTime2 = std::chrono::high_resolution_clock::now();
    result.preprocessingTime = preprocessingTime2 - preprocessingTime1;
    std::cout << "PreprocessingTime " << result.preprocessingTime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point trainingTime1 = std::chrono::high_resolution_clock::now();
    factory.TrainAllMethods();
    std::chrono::high_resolution_clock::time_point trainingTime2 = std::chrono::high_resolution_clock::now();
    result.trainingTime = trainingTime2 - trainingTime1;
    std::cout << "TrainingTime " << result.trainingTime.count() << std::endl;

    //factory.TestAllMethods();
    //factory.EvaluateAllMethods();

    reader->BookMVA("BDTG","TMVAClassification/weights/TMVAClassification_BDTG.weights.xml");
    result.probabilities.resize(test.numberOfEvents);
    
    // Apply classifier on test data
    std::chrono::high_resolution_clock::time_point testTime1 = std::chrono::high_resolution_clock::now();
    for(unsigned int iEvent = 0; iEvent < test.numberOfEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < test.numberOfFeatures; ++iFeature) {
          vec[iFeature] = test.X[iEvent][iFeature];
        }
        result.probabilities[iEvent] = (reader->EvaluateMVA("BDTG") + 1)*0.5;
    }
    std::chrono::high_resolution_clock::time_point testTime2 = std::chrono::high_resolution_clock::now();
    result.testTime = testTime2 - testTime1;
    std::cout << "TestTime " << result.testTime.count() << std::endl;

    delete reader;
    delete signal_tree;
    delete background_tree;

    return result;

}

Result measureXGBoost(const Data& train, const Data& test, const Config& config) {
    
    Result result;
    result.label = "XGBoost";

    std::chrono::high_resolution_clock::time_point preprocessingTime1 = std::chrono::high_resolution_clock::now();
    // Create XGDMatrix
    float *matrix = new float[train.numberOfEvents*train.numberOfFeatures];
    for(unsigned int iEvent = 0; iEvent < train.numberOfEvents; ++iEvent)
      for(unsigned int iFeature = 0; iFeature < train.numberOfFeatures; ++iFeature)
        matrix[iEvent*train.numberOfFeatures + iFeature] = train.X[iEvent][iFeature];

    DMatrixHandle dmatrix;
    XGDMatrixCreateFromMat(matrix, train.numberOfEvents, train.numberOfFeatures, NAN, &dmatrix);
    delete[] matrix;
    
    XGDMatrixSetUIntInfo(dmatrix, "label", train.y.data(), train.numberOfEvents);
    
    std::chrono::high_resolution_clock::time_point preprocessingTime2 = std::chrono::high_resolution_clock::now();
    result.preprocessingTime = preprocessingTime2 - preprocessingTime1;
    std::cout << "PreprocessingTime " << result.preprocessingTime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point trainingTime1 = std::chrono::high_resolution_clock::now();
    BoosterHandle booster;
    XGBoosterCreate(&dmatrix, 1, &booster);
    XGBoosterSetParam(booster, "max_depth", std::to_string(config.depth).c_str());
    XGBoosterSetParam(booster, "eta", std::to_string(config.shrinkage).c_str());
    XGBoosterSetParam(booster, "silent", std::to_string(1).c_str());
    XGBoosterSetParam(booster, "subsample", std::to_string(config.subSampling).c_str());
    XGBoosterSetParam(booster, "nthread", std::to_string(1).c_str());
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    XGBoosterSetParam(booster, "tree_method", "hist");

    // Train classifier using training data
    for(unsigned int iBoost = 0; iBoost < config.nTrees; ++iBoost) {
      XGBoosterUpdateOneIter(booster, iBoost, dmatrix);
    }

    std::chrono::high_resolution_clock::time_point trainingTime2 = std::chrono::high_resolution_clock::now();
    result.trainingTime = trainingTime2 - trainingTime1;
    std::cout << "TrainingTime " << result.trainingTime.count() << std::endl;

    result.probabilities.resize(test.numberOfEvents);
    
    // Apply classifier on test data
    std::chrono::high_resolution_clock::time_point testTime1 = std::chrono::high_resolution_clock::now();
    float *test_matrix = new float[test.numberOfEvents*test.numberOfFeatures];
    for(unsigned int iEvent = 0; iEvent < test.numberOfEvents; ++iEvent)
      for(unsigned int iFeature = 0; iFeature < test.numberOfFeatures; ++iFeature)
        test_matrix[iEvent*train.numberOfFeatures + iFeature] = test.X[iEvent][iFeature];
    DMatrixHandle test_dmatrix;
    XGDMatrixCreateFromMat(test_matrix, test.numberOfEvents, test.numberOfFeatures, NAN, &test_dmatrix);
    delete[] test_matrix;
    long unsigned int out_len;
    const float *out_result;
    XGBoosterPredict(booster, test_dmatrix, 0, 0, &out_len, &out_result);
    for(unsigned int iEvent = 0; iEvent < test.numberOfEvents; ++iEvent) {
      result.probabilities[iEvent] = out_result[iEvent];
    }
    std::chrono::high_resolution_clock::time_point testTime2 = std::chrono::high_resolution_clock::now();
    result.testTime = testTime2 - testTime1;
    std::cout << "TestTime " << result.testTime.count() << std::endl;

    XGBoosterFree(booster);
    XGDMatrixFree(dmatrix);
    XGDMatrixFree(test_dmatrix);

    return result;

}

void measure(Config &config, unsigned int id) {

  std::chrono::high_resolution_clock::time_point loadTime1 = std::chrono::high_resolution_clock::now();
  Data train("data/train.csv", config.numberOfFeatures, config.numberOfEvents);
  Data test("data/test.csv", config.numberOfFeatures, config.numberOfEvents);
  std::chrono::high_resolution_clock::time_point loadTime2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> loadTime = loadTime2 - loadTime1;
  std::cout << "LoadTime " << loadTime.count() << std::endl;

  // Repeat each measurement 5 times
  for(unsigned int i = 0; i < 5; ++i) {
    
    std::chrono::high_resolution_clock::time_point measureSKLearnTime1 = std::chrono::high_resolution_clock::now();
    Result resultSKLearn = measureSKLearn(train, test, config);
    std::chrono::high_resolution_clock::time_point measureSKLearnTime2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> measureSKLearnTime = measureSKLearnTime2 - measureSKLearnTime1;
    std::cout << "MeasureSKLearnTime " << measureSKLearnTime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point measureTMVATime1 = std::chrono::high_resolution_clock::now();
    Result resultTMVA = measureTMVA(train, test, config);
    std::chrono::high_resolution_clock::time_point measureTMVATime2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> measureTMVATime = measureTMVATime2 - measureTMVATime1;
    std::cout << "MeasureTMVATime " << measureTMVATime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point measureXGBoostTime1 = std::chrono::high_resolution_clock::now();
    Result resultXGBoost = measureXGBoost(train, test, config);
    std::chrono::high_resolution_clock::time_point measureXGBoostTime2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> measureXGBoostTime = measureXGBoostTime2 - measureXGBoostTime1;
    std::cout << "MeasureXGBoostTime " << measureXGBoostTime.count() << std::endl;

    std::chrono::high_resolution_clock::time_point measureFastBDTTime1 = std::chrono::high_resolution_clock::now();
    Result resultFastBDT = measureFastBDT(train, test, config);
    std::chrono::high_resolution_clock::time_point measureFastBDTTime2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> measureFastBDTTime = measureFastBDTTime2 - measureFastBDTTime1;
    std::cout << "MeasureFastBDTTime " << measureFastBDTTime.count() << std::endl;

    writeResults(std::string("result_") + std::to_string(id+i) + std::string("_cpp.txt"), {resultFastBDT, resultXGBoost, resultSKLearn, resultTMVA}, test, config);
  }
}

int main(int argc, char *argv[]) {

  Py_Initialize();
  import_array();

  Config config;
  config.nTrees = 100;
  config.depth = 3;
  config.shrinkage = 0.1;
  config.subSampling = 0.5;
  config.nCutLevels = 8;
  config.numberOfEvents = 800000;
  config.numberOfFeatures = 40;
 
  unsigned int id = atoi(argv[1]);

  // Here you can choose different hyper-parameters depending on the passed id
  //config.nTrees = id*10;
  //config.numberOfEvents = 500000 >> (id - 1);
  //config.numberOfFeatures = id;
  //config.depth = id;
  //config.subSampling = 0.1*id;

  measure(config, id*10);

  Py_Finalize();

}

