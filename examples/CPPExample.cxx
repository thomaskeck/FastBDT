/**
 * Thomas Keck 2016
 *
 * Using FastBDT from C++ is probably the most complicated way,
 * using python or TMVA is a lot easiert.
 * Nevertheless if you start from this example and adapt to your needs
 * you should have everything you need.
 */

#include "FastBDT.h"
#include "FastBDT_IO.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

std::vector<double> MultiGauss(const std::vector<float>&means,
                               const std::vector<float> &eigenvalues, const std::vector<std::vector<float>> &eigenvectors,
                               std::normal_distribution<double> &distribution, std::default_random_engine &generator) {

  std::vector<double> gen(means.size());

  for(unsigned int i = 0; i < means.size(); i++) {
    double variance = eigenvalues[i];
    gen[i] = sqrt(variance)*distribution(generator);
  }
  
  std::vector<double> event(means.size());
  for(unsigned int i = 0; i < means.size(); ++i)
    for(unsigned int j = 0; j < means.size(); ++j)
      event[i] = eigenvectors[i][j] * gen[j] + means[i];
  return event;


}

int main() {

    /**
     * Create MC sample, first 5 columns are the features, last column is the target
     */
   std::default_random_engine generator;
   std::normal_distribution<double> distribution(0.0,1.0);
   std::vector<float> means = {5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
   std::vector<std::vector<float>> cov = {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
																		    	{0.0, 1.0, 0.0, 0.0, 0.0, 0.1},
																		    	{0.0, 0.0, 1.0, 0.0, 0.0, 0.2},
																	    		{0.0, 0.0, 0.0, 1.0, 0.0, 0.4},
																		    	{0.0, 0.0, 0.0, 0.0, 1.0, 0.8},
																		    	{0.0, 0.1, 0.2, 0.4, 0.8, 1.0}};

    // Since I don't want to implement a matrix diagonalisation here I just state the result here,
    // so if you want to change the covariance you actually have to recalculate the eigenvalues, and vectors
    // There is some code in the TMVAExample.cxx which outputs the eigenvalues and eigenvectors
    std::vector<std::vector<float>> eigenvectors = {{0, -0, 1, 0, 0, -0, },
                                                     {-0.0766965, 0.20251, 0, 0, 0.973255, -0.0766965, },
                                                     {-0.153393, 0.0988099, 0, -0.970143, -0.0447359, -0.153393, },
                                                     {-0.306786, -0.890512, 0, 0, 0.136941, -0.306786, },
                                                     {-0.613572, 0.39524, 0, 0.242536, -0.178944, -0.613572, },
                                                     {0.707107, -2.62564e-16, 0, 0, 0, -0.707107, },
                                                    };
    std::vector<float> eigenvalues = {0.0780455, 1, 1, 1, 1, 1.92195, };
    
    std::vector<std::vector<double>> data(10000);
    for(unsigned int iEvent = 0; iEvent < 10000; ++iEvent) {
        std::vector<double> event = MultiGauss(means, eigenvalues, eigenvectors, distribution, generator);
        event[5] = (event[5] > 0.0) ? 1.0 : 0.0;
        data[iEvent] = event;
    }



    std::chrono::high_resolution_clock::time_point measureTTime1 = std::chrono::high_resolution_clock::now();
    /*
     * The FastBDT expects binned data! You can use a FeatureBinning for each feature to bin your data.
     * The binning is an equal statistics binning. So the shape of the input data doesn't matter! (So noramlisation or gaussianisation is not necessary)
     *
     * Basically you just create the FeatureBinning using its constructor:
     *  First argument is the 'binning-level'. In total 2^n bins are used for the feature, in this case 2^4 = 16 (all FeatureBinnings need the SAME binning-level)
     *  Second argument is an STL container
     */
    std::vector<FastBDT::FeatureBinning<double>> featureBinnings;
    for(unsigned int iFeature = 0; iFeature < 5; ++iFeature) {
        std::vector<double> feature;
        feature.reserve(data.size());
        for(auto &event : data) {
            feature.push_back( event[iFeature] );
        }
        featureBinnings.push_back(FastBDT::FeatureBinning<double>(8, feature));
    }
    
    /**
     * FastBDT expects the input data in its own format called EventSample,
     * the EventSample automatically sorts the data into signal and background and allows
     * an optimized and fast access during the training.
     *
     * EventSample needs the total size of the training data, the number of features and the binning-level of the FeatureBinnings.
     * In the example below the data is now binned using the FeatureBinning's ValueToBin method
     * and afterwards added (each row seperatly) into the EventSample using its AddEvent method.
     * Second parameter is the number of features
     * Third parameter is the number of binning levels for each feature
     */
    FastBDT::EventSample eventSample(data.size(), 5, {8, 8, 8, 8, 8});
    for(auto &event : data) {
        int _class = int(event.back());
        std::vector<unsigned int> bins(5);
        for(unsigned int iFeature = 0; iFeature < 5; ++iFeature) {
            bins[iFeature] = featureBinnings[iFeature].ValueToBin( event[iFeature] );
        }
        // First argument is the event data as a std::vector,
        // second argument is the event weight
        // third argument is a bool, true indicates signal, false background
        eventSample.AddEvent(bins, 1.0, _class == 1);
    }
    
    /**
     * Now the aglorithm is trained via the ForestBuilder,
     * the arguments are nTrees, shrinkage, randRatio and treeDepth
     */
    FastBDT::ForestBuilder dt(eventSample, 200, 0.1, 0.5, 3);

    /**
     * To apply the trained method again we create a Forest using the trees
     * from the ForestBuilder
     */
    FastBDT::Forest<double> forest( dt.GetShrinkage(), dt.GetF0(), true);
    for( auto t : dt.GetForest() )
        /**
         * Here we remove the binning from the tree, so we don't have to apply it during the application
         * phase, this is not necessary, but it has only advantages (speed) and no drawbacks
         */
        forest.AddTree(FastBDT::removeFeatureBinningTransformationFromTree(t, featureBinnings));
    
    std::chrono::high_resolution_clock::time_point measureTTime2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> measureTTime = measureTTime2 - measureTTime1;
    std::cout << "Finished training in " << measureTTime.count() << " ms " << std::endl;
    
    /**
     * Saving featureBinnings and forest into a file:
     */
    std::fstream file("example_forest.weightfile", std::ios_base::out | std::ios_base::trunc);
    file << forest << std::endl;
    file.close();
    
    // We can read in the forest frm the file later using this
    // auto forest = FastBDT::readForestFromStream<double>(file);
       
    /*
     * We the tree to new data (or in this case just the training sample)
     * using the Analyse Function which will return something like a probability between 0 and 1
     * We don't need to apply the binning here, because we remved the binning transformation from the tree
     * earlier, if we hadn't done this, we would need to provide binned data again!
     */
    std::chrono::high_resolution_clock::time_point measureATime1 = std::chrono::high_resolution_clock::now();
    unsigned int correct = 0;
    for(auto &event : data) {
        int _class = int(event.back());
        double p = forest.Analyse(event);
        if (_class == 1 and p > 0.5 or _class == 0 and p <= 0.5) {
            correct++;
        }
    }
    std::chrono::high_resolution_clock::time_point measureATime2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> measureATime = measureATime2 - measureATime1;
    std::cout << "Finished application in " << measureATime.count() << " ms " << std::endl;

    std::cout << "The forest classified " << correct / static_cast<float>(data.size()) << " % of the samples correctly" << std::endl;

    return 0;
}
