/**
 * Thomas Keck 2015
 */

#include "FastBDT_C_API.h"

#include <fstream>
#include <new>
#include <iostream>

using namespace FastBDT;

extern "C" {

    void PrintVersion() {
      std::cerr << "FastBDT Version: " << FastBDT_VERSION_MAJOR << "." << FastBDT_VERSION_MINOR << std::endl;
    }

    void* Create() {
      Expertise *expertise = new(std::nothrow) Expertise;
      return expertise;
    }
    
    void SetBinning(void *ptr, unsigned int* binning, unsigned int size) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetBinning(std::vector<unsigned int>(binning, binning + size));
    }

    void SetPurityTransformation(void *ptr, bool* purityTransformation, unsigned int size) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetPurityTransformation(std::vector<bool>(purityTransformation, purityTransformation + size));
    }
    
    void SetNTrees(void *ptr, unsigned int nTrees) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetNTrees(nTrees);
    }

    unsigned int GetNTrees(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetNTrees();
    }
    
    void SetDepth(void *ptr, unsigned int depth) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetDepth(depth);
    }

    unsigned int GetDepth(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetDepth();
    }
    
    void SetNumberOfFlatnessFeatures(void *ptr, unsigned int numberOfFlatnessFeatures) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetNumberOfFlatnessFeatures(numberOfFlatnessFeatures);
    }

    unsigned int GetNumberOfFlatnessFeatures(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetNumberOfFlatnessFeatures();
    }
    
    void SetSubsample(void *ptr, double subsample) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetSubsample(subsample);
    }

    double GetSubsample(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetSubsample();
    }
    
    void SetShrinkage(void *ptr, double shrinkage) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetShrinkage(shrinkage);
    }

    double GetShrinkage(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetShrinkage();
    }
    
    void SetFlatnessLoss(void *ptr, double flatnessLoss) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetFlatnessLoss(flatnessLoss);;
    }

    double GetFlatnessLoss(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetFlatnessLoss();
    }

    void SetTransform2Probability(void *ptr, bool transform2probability) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetTransform2Probability(transform2probability);
    }

    bool GetTransform2Probability(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetTransform2Probability();
    }
    
    void SetSPlot(void *ptr, bool sPlot) {
      reinterpret_cast<Expertise*>(ptr)->classifier.SetSPlot(sPlot);
    }

    bool GetSPlot(void *ptr) {
      return reinterpret_cast<Expertise*>(ptr)->classifier.GetSPlot();
    }

    void Delete(void *ptr) {
      delete reinterpret_cast<Expertise*>(ptr);
    }
    
    void Fit(void *ptr, float *data_ptr, float *weight_ptr, bool *target_ptr, unsigned int nEvents, unsigned int nFeatures) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      std::vector<float> w;
      if(weight_ptr != nullptr)
        w = std::vector<float>(weight_ptr, weight_ptr + nEvents);
      else
        w = std::vector<float>(nEvents, 1.0);

      std::vector<bool> y(target_ptr, target_ptr + nEvents);
      std::vector<std::vector<float>> X(nFeatures);
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
        std::vector<float> temp(nEvents);
        for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
          temp[iEvent] = data_ptr[iEvent*nFeatures + iFeature];
        }
        X[iFeature] = temp;
      }

      expertise->classifier.fit(X, y, w);

    }

    void Load(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      
      std::fstream file(weightfile, std::ios_base::in);
      if(not file)
    	  return;

      expertise->classifier = FastBDT::Classifier(file);
    }

    float Predict(void *ptr, float *array) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      return expertise->classifier.predict(std::vector<float>(array, array + expertise->classifier.GetNFeatures()));
    }
    
    void PredictArray(void *ptr, float *array, float *result, unsigned int nEvents) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      unsigned int nFeatures = expertise->classifier.GetNFeatures();
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
        result[iEvent] = expertise->classifier.predict(std::vector<float>(array + iEvent*nFeatures, array + (iEvent+1)*nFeatures));
      }
    }

    void Save(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      std::fstream file(weightfile, std::ios_base::out | std::ios_base::trunc);
      file << expertise->classifier << std::endl;
    }
  
    void* GetVariableRanking(void* ptr) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      VariableRanking *ranking = new(std::nothrow) VariableRanking;
      ranking->ranking = expertise->classifier.GetVariableRanking();
      return ranking;
    }
    
    void* GetIndividualVariableRanking(void* ptr, float *array) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      VariableRanking *ranking = new(std::nothrow) VariableRanking;
      ranking->ranking = expertise->classifier.GetIndividualVariableRanking(std::vector<float>(array, array + expertise->classifier.GetNFeatures()));
      return ranking;
    }
    
    unsigned int ExtractNumberOfVariablesFromVariableRanking(void* ptr) {
      VariableRanking *ranking = reinterpret_cast<VariableRanking*>(ptr);
      unsigned int max = 0;
      for(auto &pair : ranking->ranking) {
        if(pair.first > max) {
          max = pair.first;
        }
      }
      return max+1;
    }
    
    double ExtractImportanceOfVariableFromVariableRanking(void* ptr, unsigned int iFeature) {
      VariableRanking *ranking = reinterpret_cast<VariableRanking*>(ptr);
      if ( ranking->ranking.find( iFeature ) == ranking->ranking.end() )
        return 0.0;
      return ranking->ranking[iFeature];  
    }
    
    void DeleteVariableRanking(void *ptr) {
      delete reinterpret_cast<VariableRanking*>(ptr);
    }

}
