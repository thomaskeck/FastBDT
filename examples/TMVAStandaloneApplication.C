/**
 * Thomas Keck 2016
 *
 * Uses TMVA Standalone C++ class, which is written out by
 * the TMVAExample.C
 *
 * It does not depend on FastBDT, ROOT or anything other than
 * the standard libaries!
 *
 * Compile this after
 * root -l examples/TMVAExample.C
 *
 * using
 * g++ examples/TMVAStandaloneApplication.C -o test -I.
 * ./test
 */


#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>

#include "weights/TMVAClassification_FastBDT.class.C"

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
        // Reverse features so that they have the same order as in the TMVA example
        std::reverse(event.begin(), event.begin()+5);
        data[iEvent] = event;
    }

  std::vector<std::string> features = { "FeatureA", "FeatureB", "FeatureC", "FeatureD", "FeatureE" };
  ReadFastBDT fbdt(features);
    
  std::chrono::high_resolution_clock::time_point measureATime1 = std::chrono::high_resolution_clock::now();
  unsigned int correct = 0;
  for(auto &event : data) {
      int _class = int(event.back());
      double p = fbdt.GetMvaValue(event);
      if (_class == 1 and p > 0.5 or _class == 0 and p <= 0.5) {
          correct++;
      }
  }
  std::chrono::high_resolution_clock::time_point measureATime2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> measureATime = measureATime2 - measureATime1;
  std::cout << "Finished application in " << measureATime.count() << " ms " << std::endl;

  std::cout << "The forest classified " << correct / static_cast<float>(data.size()) << " % of the samples correctly" << std::endl;


}
