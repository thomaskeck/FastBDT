/**
 * Thomas Keck 2015
 */

#include "FastBDT.h"
#include "IO.h"

extern "C" {

    void PrintVersion();

    struct Expertise {
      std::vector<FastBDT::FeatureBinning<double>> featureBinnings;
      FastBDT::Forest forest;
      unsigned int nBinningLevels;
      unsigned int nTrees;
      double shrinkage;
      double randRatio;
      unsigned int nLayersPerTree;
    };

    void* Create();

    void SetNBinningLevels(void *ptr, unsigned int nBinningLevels);
    
    void SetNTrees(void *ptr, unsigned int nTrees);
    
    void SetNLayersPerTree(void *ptr, unsigned int nLayersPerTree);
    
    void SetShrinkage(void *ptr, double shrinkage);
    
    void SetRandRatio(void *ptr, double randRatio);

    void Delete(void *ptr);
    
    void Train(void *ptr, void *data_ptr, void *target_ptr, unsigned int nEvents, unsigned int nFeatures);

    void Load(void* ptr, char *weightfile);

    double Analyse(void *ptr, double *array);

    void Save(void* ptr, char *weightfile);

}
