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
      expertise->nBinningLevels = 8;
      expertise->nTrees = 100;
      expertise->shrinkage = 0.1;
      expertise->randRatio = 0.5;
      expertise->nLayersPerTree = 3;
      expertise->transform2probability = true;
      return expertise;
    }

    void SetNBinningLevels(void *ptr, unsigned int nBinningLevels) {
      reinterpret_cast<Expertise*>(ptr)->nBinningLevels = nBinningLevels;
    }
    
    void SetNTrees(void *ptr, unsigned int nTrees) {
      reinterpret_cast<Expertise*>(ptr)->nTrees = nTrees;
    }
    
    void SetNLayersPerTree(void *ptr, unsigned int nLayersPerTree) {
      reinterpret_cast<Expertise*>(ptr)->nLayersPerTree = nLayersPerTree;
    }
    
    void SetShrinkage(void *ptr, double shrinkage) {
      reinterpret_cast<Expertise*>(ptr)->shrinkage = shrinkage;
    }
    
    void SetRandRatio(void *ptr, double randRatio) {
      reinterpret_cast<Expertise*>(ptr)->randRatio = randRatio;
    }
    
    void SetTransform2Probability(void *ptr, bool transform2probability) {
      reinterpret_cast<Expertise*>(ptr)->transform2probability = transform2probability;
    }

    void Delete(void *ptr) {
      delete reinterpret_cast<Expertise*>(ptr);
    }
    
    void Train(void *ptr, void *data_ptr, void *weight_ptr, void *target_ptr, unsigned int nEvents, unsigned int nFeatures) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      double *data = reinterpret_cast<double*>(data_ptr);
      float *weights = reinterpret_cast<float*>(weight_ptr);
      unsigned int *target = reinterpret_cast<unsigned int*>(target_ptr);

      std::vector<unsigned int> nLevels;
      std::vector<FastBDT::FeatureBinning<double>> featureBinnings;
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
        std::vector<double> feature(nEvents);
        for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
          feature[iEvent] = *(data + iEvent*nFeatures + iFeature);
        }
        featureBinnings.push_back(FeatureBinning<double>(expertise->nBinningLevels, feature));
        nLevels.push_back(expertise->nBinningLevels);
      }

      EventSample eventSample(nEvents, nFeatures, nLevels);
      std::vector<unsigned int> bins(nFeatures);
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
          bins[iFeature] = featureBinnings[iFeature].ValueToBin(data[iEvent*nFeatures + iFeature]);
        }
        eventSample.AddEvent(bins, (weight_ptr != nullptr) ? weights[iEvent] : 1.0, target[iEvent] == 1);
      }

      ForestBuilder df(eventSample, expertise->nTrees, expertise->shrinkage, expertise->randRatio, expertise->nLayersPerTree);
      Forest<double> forest( df.GetShrinkage(), df.GetF0(), expertise->transform2probability);
      for( auto t : df.GetForest() ) {
         forest.AddTree(removeFeatureBinningTransformationFromTree(t, featureBinnings));
      }
      expertise->forest = forest;

    }

    void Load(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      
      std::fstream file(weightfile, std::ios_base::in);
      if(not file)
    	  return;

      expertise->forest = FastBDT::readForestFromStream<double>(file);
    }

    double Analyse(void *ptr, double *array) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      return expertise->forest.Analyse(array);
    }
    
    void AnalyseArray(void *ptr, double *array, double *result, unsigned int nEvents, unsigned int nFeatures) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
        result[iEvent] = expertise->forest.Analyse(&array[iEvent*nFeatures]);
      }
    }

    void Save(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      std::fstream file(weightfile, std::ios_base::out | std::ios_base::trunc);
      file << expertise->forest << std::endl;
    }
  
    void* GetVariableRanking(void* ptr) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      VariableRanking *ranking = new(std::nothrow) VariableRanking;
      ranking->ranking = expertise->forest.GetVariableRanking();
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
