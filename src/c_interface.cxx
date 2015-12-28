/**
 * Thomas Keck 2015
 */

#include "c_interface.h"

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

    void Delete(void *ptr) {
      delete reinterpret_cast<Expertise*>(ptr);
    }
    
    void Train(void *ptr, void *data_ptr, void *target_ptr, unsigned int nEvents, unsigned int nFeatures) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      double *data = reinterpret_cast<double*>(data_ptr);
      unsigned int *target = reinterpret_cast<unsigned int*>(target_ptr);

      std::vector<unsigned int> nLevels;
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
        std::vector<double> feature(nEvents);
        for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
          feature[iEvent] = *(data + iEvent*nFeatures + iFeature);
        }
        expertise->featureBinnings.push_back(FeatureBinning<double>(expertise->nBinningLevels, feature.begin(), feature.end()));
        nLevels.push_back(expertise->nBinningLevels);
      }

      EventSample eventSample(nEvents, nFeatures, nLevels);
      std::vector<unsigned int> bins(nFeatures);
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
          bins[iFeature] = expertise->featureBinnings[iFeature].ValueToBin(data[iEvent*nFeatures + iFeature]);
        }
        eventSample.AddEvent(bins, 1.0, target[iEvent] == 1);
      }

      ForestBuilder df(eventSample, expertise->nTrees, expertise->shrinkage, expertise->randRatio, expertise->nLayersPerTree);
      expertise->forest = Forest( df.GetShrinkage(), df.GetF0());
      for( auto t : df.GetForest() ) {
          expertise->forest.AddTree(t);
      }

    }

    void Load(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      
      std::fstream file(weightfile, std::ios_base::in);
      if(not file)
    	  return;

      file >> expertise->featureBinnings;
      expertise->forest = FastBDT::readForestFromStream(file);
    }

    double Analyse(void *ptr, double *array) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      unsigned int numberOfFeatures = expertise->featureBinnings.size();
      std::vector<unsigned int> bins(numberOfFeatures);
      for(unsigned int iFeature = 0; iFeature < numberOfFeatures; ++iFeature) {
        bins[iFeature] = expertise->featureBinnings[iFeature].ValueToBin(array[iFeature]);
      }

      return expertise->forest.Analyse(bins);
    }

    void Save(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      std::fstream file(weightfile, std::ios_base::out | std::ios_base::trunc);
      file << expertise->featureBinnings << std::endl;
      file << expertise->forest << std::endl;
    }

}
