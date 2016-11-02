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
      expertise->purityTransformation = 0;
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
    
    void SetPurityTransformation(void *ptr, unsigned int purityTransformation) {
      reinterpret_cast<Expertise*>(ptr)->purityTransformation = purityTransformation;
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
      expertise->featureBinnings.clear();
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
        std::vector<double> feature(nEvents);
        for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
          feature[iEvent] = *(data + iEvent*nFeatures + iFeature);
        }
        expertise->featureBinnings.push_back(FeatureBinning<double>(expertise->nBinningLevels, feature));
        nLevels.push_back(expertise->nBinningLevels);
      }
          
      unsigned int nFeaturesFinal = nFeatures;
      if( expertise->purityTransformation > 0 ) {
          if( expertise->purityTransformation == 2 ) {
            nFeaturesFinal = 2*nFeatures;
          }
          std::vector<float> v_weights(nEvents, 1.0);
          if(weights != nullptr)
            v_weights.assign(weights, weights + nEvents);
          std::vector<bool> v_isSignal(nEvents);
          for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
            v_isSignal[iEvent] = target[iEvent] == 1;
          }
          for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
            std::vector<unsigned int> feature(nEvents);
            for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
              feature[iEvent] = expertise->featureBinnings[iFeature].ValueToBin(data[iEvent*nFeatures + iFeature]);
            }
            expertise->purityTransformations.push_back(PurityTransformation(expertise->nBinningLevels, feature, v_weights, v_isSignal));
          }
      }

      EventSample eventSample(nEvents, nFeaturesFinal, nLevels);
      std::vector<unsigned int> bins(nFeaturesFinal);
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
        for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
          bins[iFeature] = expertise->featureBinnings[iFeature].ValueToBin(data[iEvent*nFeatures + iFeature]);
          if( expertise->purityTransformation == 1 ) {
              bins[iFeature] = expertise->purityTransformations[iFeature].BinToPurityBin(bins[iFeature]);
          } else if (expertise->purityTransformation == 2) {
              bins[iFeature + nFeatures] = expertise->purityTransformations[iFeature].BinToPurityBin(bins[iFeature]);
          }
        }
        eventSample.AddEvent(bins, (weight_ptr != nullptr) ? weights[iEvent] : 1.0, target[iEvent] == 1);
      }

      ForestBuilder df(eventSample, expertise->nTrees, expertise->shrinkage, expertise->randRatio, expertise->nLayersPerTree);
      if( expertise->purityTransformation > 0) {
          Forest<unsigned int> forest( df.GetShrinkage(), df.GetF0(), expertise->transform2probability);
          for( auto t : df.GetForest() ) {
             forest.AddTree(t);
          }
          expertise->binned_forest = forest;
      } else {
          Forest<double> forest( df.GetShrinkage(), df.GetF0(), expertise->transform2probability);
          for( auto t : df.GetForest() ) {
             forest.AddTree(removeFeatureBinningTransformationFromTree(t, expertise->featureBinnings));
          }
          expertise->forest = forest;
      }
    }

    void Load(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      
      std::fstream file(weightfile, std::ios_base::in);
      if(not file)
    	  return;

      expertise->forest = FastBDT::readForestFromStream<double>(file);
      expertise->binned_forest = FastBDT::readForestFromStream<unsigned int>(file);
      file >> expertise->featureBinnings;
      file >> expertise->purityTransformations;
      file >> expertise->purityTransformation;

    }

    double Analyse(void *ptr, double *array) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      if(expertise->purityTransformation > 0) {
        unsigned int nFeatures = expertise->purityTransformations.size();
        unsigned int nFeaturesFinal = nFeatures;
        if( expertise->purityTransformation == 2 ) {
            nFeaturesFinal *= 2;
        }    
        std::vector<unsigned int> bins(nFeaturesFinal);
        for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
          bins[iFeature] = expertise->featureBinnings[iFeature].ValueToBin(array[iFeature]);
          if( expertise->purityTransformation == 1 ) {
              bins[iFeature] = expertise->purityTransformations[iFeature].BinToPurityBin(bins[iFeature]);
          } else if ( expertise->purityTransformation == 2 ) {
              bins[iFeature + nFeatures] = expertise->purityTransformations[iFeature].BinToPurityBin(bins[iFeature]);
          }
        }
        return expertise->binned_forest.Analyse(bins);
      } else {
        return expertise->forest.Analyse(array);
      }
    }
    
    void AnalyseArray(void *ptr, double *array, double *result, unsigned int nEvents, unsigned int nFeatures) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      if(expertise->purityTransformation > 0) {
          unsigned int nFeatures = expertise->purityTransformations.size();
          unsigned int nFeaturesFinal = nFeatures;
          if( expertise->purityTransformation == 2 ) {
              nFeaturesFinal *= 2;
          }    
          std::vector<unsigned int> bins(nFeaturesFinal);
          for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
            for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
              bins[iFeature] = expertise->featureBinnings[iFeature].ValueToBin(array[iEvent*nFeatures + iFeature]);
              if( expertise->purityTransformation == 1 ) {
                  bins[iFeature] = expertise->purityTransformations[iFeature].BinToPurityBin(bins[iFeature]);
              } else if ( expertise->purityTransformation == 2 ) {
                  bins[iFeature + nFeatures] = expertise->purityTransformations[iFeature].BinToPurityBin(bins[iFeature]);
              }
            }
            result[iEvent] = expertise->binned_forest.Analyse(bins);
          }
      } else {
          for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
            result[iEvent] = expertise->forest.Analyse(&array[iEvent*nFeatures]);
          }
      }
    }

    void Save(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      std::fstream file(weightfile, std::ios_base::out | std::ios_base::trunc);
      file << expertise->forest << std::endl;
      file << expertise->binned_forest << std::endl;
      file << expertise->featureBinnings << std::endl;
      file << expertise->purityTransformations << std::endl;
      file << expertise->purityTransformation << std::endl;
    }
  
    void* GetVariableRanking(void* ptr) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      VariableRanking *ranking = new(std::nothrow) VariableRanking;
      ranking->ranking = expertise->forest.GetVariableRanking();
      return ranking;
    }
    
    void* GetIndividualVariableRanking(void* ptr, double *array) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      VariableRanking *ranking = new(std::nothrow) VariableRanking;
      ranking->ranking = expertise->forest.GetIndividualVariableRanking(array);
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
