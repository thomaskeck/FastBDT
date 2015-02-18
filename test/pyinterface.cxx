#include "FBDT.h"
#include <new>
#include <iostream>

using namespace FastBDT;

extern "C" {


    void* CreateFeatureBinning(int nBinningLevels, int nEvents, float *array) {
        return new(std::nothrow) FeatureBinning<float>(static_cast<unsigned int>(nBinningLevels), array, array+nEvents);
    }

    void DeleteFeatureBinning(void *ptr) {
        delete reinterpret_cast<FeatureBinning<float> *>(ptr);
    }

    int FeatureBinningValueToBin(void *ptr, float value) {
        return static_cast<int>( reinterpret_cast<FeatureBinning<float> *>(ptr)->ValueToBin(value));
    }

    void* CreateEventSample(int nEvents, int nFeatures, int nBinningLevels) {
        return new(std::nothrow) EventSample(static_cast<unsigned int>(nEvents), static_cast<unsigned int>(nFeatures) , static_cast<unsigned int>(nBinningLevels));
    }
    
    void DeleteEventSample(void *ptr) {
        delete reinterpret_cast<EventSample*>(ptr);
    }

    void EventSampleAddEvent(void *ptr, int nFeatures, int *c_bins, int _class) {
        std::vector<unsigned int> bins;
        bins.assign(c_bins, c_bins + nFeatures);
        /*std::cout << _class << std::endl;
        for(int i = 0; i < nFeatures; ++i)
          std::cout << bins[i] << " ";
        std::cout << std::endl;*/
        reinterpret_cast<EventSample*>(ptr)->AddEvent(bins, 1.0, _class == 1);
    }

    void* CreateForest(void *eventSample, int nTrees, double shrinkage, double randRatio, int nLayersPerTree) {

        ForestBuilder df(*reinterpret_cast<EventSample*>(eventSample), static_cast<unsigned int>(nTrees), shrinkage, randRatio, static_cast<unsigned int>(nLayersPerTree));
        //df.GetForest()[0].Print();   
        Forest *forest = new(std::nothrow) Forest( df.GetShrinkage(), df.GetF0());
        for( auto t : df.GetForest() ) {
            forest->AddTree(t);
        }
        return forest;

    }
    
    void DeleteForest(void *ptr) {
        delete reinterpret_cast<Forest*>(ptr);
    }

    double ForestAnalyse(void *ptr, int *bins) {
        return reinterpret_cast<Forest*>(ptr)->Analyse(bins);
    }

}
