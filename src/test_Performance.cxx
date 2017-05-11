/**
 * Thomas Keck 2017
 */

#include "FastBDT.h"

#include <gtest/gtest.h>

#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>

using namespace FastBDT;

class PerformanceFeatureBinningTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            std::default_random_engine generator;
            std::uniform_real_distribution<double> distribution(0.0,1.0);
            unsigned int N = 10000000;
            data.resize(N);
            for(unsigned int i = 0; i < N; ++i) {
              data[i] = distribution(generator);
            }
        }

        std::vector<float> data;

};


TEST_F(PerformanceFeatureBinningTest, FeatureBinningScalesLinearInNumberOfDataPoints) {

    // This is dominated by the sorting of the numbers -> N log (N),
    // for our purposes we assume just N, which seems to be fine
    // if this unittest starts failing I have to revise this and add the factor of log(N)

    std::vector<unsigned int> sizes = {1000, 10000, 100000, 1000000};
    std::vector<double> times;

    for( auto &size : sizes ) {
      std::vector<float> temp_data(data.begin(), data.begin() + size);
      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
      FeatureBinning<float> binning(4, temp_data);
      std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

      // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
      EXPECT_EQ(binning.GetNLevels(), 4u);

      std::chrono::duration<double, std::micro> time = stop - start;
      times.push_back(time.count());
    }

    // Check linear behaviour
    for(unsigned int i = 1; i < sizes.size(); ++i) {
      double size_ratio = sizes[i] / static_cast<double>(sizes[0]);
      double time_ratio = times[i] / static_cast<double>(times[0]);
      // We allow for deviation of factor two
      EXPECT_LT(time_ratio,  size_ratio * 2.0);
    }

}


TEST_F(PerformanceFeatureBinningTest, FeatureBinningScalesConstantInSmallNumberOfLayers) {

    // The feature binning should be dominated by the sorting of the numbers
    // hence it does not scale with the number of layers to first order
    // for large layers this will be wrong ~ #Layer > 17
    std::vector<unsigned int> sizes = {2, 3, 5, 7, 11, 13, 17};
    std::vector<double> times;

    for( auto &size : sizes ) {
      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
      FeatureBinning<float> binning(size, data);
      std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

      // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
      EXPECT_EQ(binning.GetNLevels(), size);

      std::chrono::duration<double, std::micro> time = stop - start;
      times.push_back(time.count());
    }

    // Check linear behaviour
    // We ignore the first measurement, to avoids effects of caching
    for(unsigned int i = 2; i < sizes.size(); ++i) {
      double time_ratio = times[i] / static_cast<double>(times[1]);
      EXPECT_GT(time_ratio,  0.8);
      EXPECT_LT(time_ratio,  1.2);
    }

}

class PerformanceTreeBuilderTest : public ::testing::Test {
    protected:
        std::default_random_engine generator;
        std::uniform_int_distribution<unsigned int> distribution{0, 16};
};

TEST_F(PerformanceTreeBuilderTest, TreeBuilderScalesLinearInNumberOfDataPoints) {

    auto random_source = std::bind(distribution, generator);

    unsigned int nFeatures = 10;
    unsigned int nLayers = 4;
    
    std::vector<unsigned int> sizes = {1000, 10000, 100000, 1000000, 10000000};
    std::vector<double> times;

    for( auto &size : sizes ) {
      unsigned int nDataPoints = size;
      std::vector<unsigned int> row(nFeatures);
      std::vector<unsigned int> binning_levels(nFeatures, 4);

      EventSample sample(nDataPoints, nFeatures, 0, binning_levels);
      for(unsigned int i = 0; i < nDataPoints; ++i) {
        std::generate_n(row.begin(), nFeatures, random_source); 
        sample.AddEvent( row, 1.0, i % 2 == 0);
      }

      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
      TreeBuilder dt(nLayers, sample);
      std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

      // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
      const auto &purities = dt.GetPurities();
      EXPECT_EQ(purities.size(), static_cast<unsigned int>((1 << (nLayers+1)) - 1));

      std::chrono::duration<double, std::micro> time = stop - start;
      times.push_back(time.count());
    }

    // Check linear behaviour
    for(unsigned int i = 1; i < sizes.size(); ++i) {
      double size_ratio = sizes[i] / static_cast<double>(sizes[0]);
      double time_ratio = times[i] / static_cast<double>(times[0]);
      // We allow for deviation of factor two
      EXPECT_LT(time_ratio,  size_ratio * 2.0);
    }


}

TEST_F(PerformanceTreeBuilderTest, TreeBuilderScalesLinearInNumberOfFeatures) {

    auto random_source = std::bind(distribution, generator);

    unsigned int nLayers = 4;
    unsigned int nDataPoints = 100000;
    
    std::vector<unsigned int> sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::vector<double> times;

    for( auto &size : sizes ) {
      unsigned int nFeatures = size;
      std::vector<unsigned int> row(nFeatures);
      std::vector<unsigned int> binning_levels(nFeatures, 4);

      EventSample sample(nDataPoints, nFeatures, 0, binning_levels);
      for(unsigned int i = 0; i < nDataPoints; ++i) {
        std::generate_n(row.begin(), nFeatures, random_source); 
        sample.AddEvent( row, 1.0, i % 2 == 0);
      }

      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
      TreeBuilder dt(nLayers, sample);
      std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

      // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
      const auto &purities = dt.GetPurities();
      EXPECT_EQ(purities.size(), static_cast<unsigned int>((1 << (nLayers+1)) - 1));

      std::chrono::duration<double, std::micro> time = stop - start;
      times.push_back(time.count());
    }

    // Check linear behaviour
    // We ignore the first measurement, to avoids effects of caching
    for(unsigned int i = 2; i < sizes.size(); ++i) {
      double size_ratio = sizes[i] / static_cast<double>(sizes[1]);
      double time_ratio = times[i] / static_cast<double>(times[1]);
      // We allow for deviation of factor two
      EXPECT_LT(time_ratio,  size_ratio * 2.0);
    }
}


TEST_F(PerformanceTreeBuilderTest, TreeBuilderScalesLinearForSmallNumberOfLayers) {

    // For small numbers of layers (below 10) we should scale linear,
    // above the number of nodes in the deeper layers of the tree gets in the same order
    // of magnitude as the number of data_points and the summing of the histograms
    // becomes important
    auto random_source = std::bind(distribution, generator);

    unsigned int nFeatures = 10;
    unsigned int nDataPoints = 100000;
    
    std::vector<unsigned int> sizes = {1, 2, 3, 5, 7, 11, 13};
    std::vector<double> times;
      
    std::vector<unsigned int> row(nFeatures);
    std::vector<unsigned int> binning_levels(nFeatures, 4);
    EventSample sample(nDataPoints, nFeatures, 0, binning_levels);
    for(unsigned int i = 0; i < nDataPoints; ++i) {
      std::generate_n(row.begin(), nFeatures, random_source); 
      sample.AddEvent( row, 1.0, i % 2 == 0);
    }

    for( auto &size : sizes ) {
      unsigned int nLayers = size;

      // Reset flags, so we can use the sample multiple times
      auto &flags = sample.GetFlags();
      for(unsigned int iEvent = 0; iEvent < nDataPoints; ++iEvent)
        flags.Set(iEvent, 1);

      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
      TreeBuilder dt(nLayers, sample);
      std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

      // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
      const auto &purities = dt.GetPurities();
      EXPECT_EQ(purities.size(), static_cast<unsigned int>((1 << (nLayers+1)) - 1));

      std::chrono::duration<double, std::micro> time = stop - start;
      times.push_back(time.count());
    }

    // Check linear behaviour
    // We ignore the first measurement, to avoids effects of caching
    for(unsigned int i = 2; i < sizes.size(); ++i) {
      double size_ratio = sizes[i] / static_cast<double>(sizes[1]);
      double time_ratio = times[i] / static_cast<double>(times[1]);
      // We allow for deviation of factor two
      EXPECT_LT(time_ratio,  size_ratio * 2.0);
    }
}
