/**
 * Thomas Keck 2015
 */

#include "FastBDT.h"
#include "FastBDT_IO.h"
#include "Classifier.h"

extern "C" {

    void PrintVersion();

    struct Expertise {
      FastBDT::Classifier classifier;
    };
      
    void* Create();

    void SetBinning(void *ptr, unsigned int* binning, unsigned int size);
    void SetPurityTransformation(void *ptr, bool* purityTransformation, unsigned int size);
    
    void SetNTrees(void *ptr, unsigned int nTrees);
    unsigned int GetNTrees(void *ptr);
    
    void SetDepth(void *ptr, unsigned int depth);
    unsigned int GetDepth(void *ptr);
    
    void SetNumberOfFlatnessFeatures(void *ptr, unsigned int numberOfFlatnessFeatures);
    unsigned int GetNumberOfFlatnessFeatures(void *ptr);
    
    void SetSubsample(void *ptr, double subsample);
    double GetSubsample(void *ptr);
    
    void SetShrinkage(void *ptr, double shrinkage);
    double GetShrinkage(void *ptr);
    
    void SetFlatnessLoss(void *ptr, double flatnessLoss);
    double GetFlatnessLoss(void *ptr);

    void SetTransform2Probability(void *ptr, bool transform2probability);
    bool GetTransform2Probability(void *ptr);
    
    void SetSPlot(void *ptr, bool sPlot);
    bool GetSPlot(void *ptr);
    
    void Delete(void *ptr);
    
    void Fit(void *ptr, float *data_ptr, float *weight_ptr, bool *target_ptr, unsigned int nEvents, unsigned int nFeatures);

    void Load(void* ptr, char *weightfile);

    float Predict(void *ptr, float *array);

    void PredictArray(void *ptr, float *array, float *result, unsigned int nEvents);

    void Save(void* ptr, char *weightfile);
    
    struct VariableRanking {
        std::map<unsigned int, double> ranking;
    }; 

    void* GetVariableRanking(void* ptr);
    
    void* GetIndividualVariableRanking(void* ptr, float *array);
    
    unsigned int ExtractNumberOfVariablesFromVariableRanking(void* ptr);
    
    double ExtractImportanceOfVariableFromVariableRanking(void* ptr, unsigned int iFeature);
    
    void DeleteVariableRanking(void* ptr);

}
