/**
 * Thomas Keck 2014
 */

#include "FastBDT.h"

#include <gtest/gtest.h>

#include <sstream>
#include <limits>

using namespace FastBDT;

class FeatureBinningTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            std::vector<float> data = {10.0f,8.0f,2.0f,NAN,NAN,NAN,NAN,7.0f,5.0f,6.0f,9.0f,NAN,4.0f,3.0f,11.0f,12.0f,1.0f,NAN};
            calculatedBinning = new FeatureBinning<float>(2, data);

            binning = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
            predefinedBinning = new FeatureBinning<float>(2, binning);
            
            // Set the binning again, because it is sorted inside the constructor
            binning = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
        }

        virtual void TearDown() {
            delete calculatedBinning;
            delete predefinedBinning;
        }

        unsigned int nLevels;
        std::vector<float> binning;
        FeatureBinning<float> *calculatedBinning;
        FeatureBinning<float> *predefinedBinning;

};

TEST_F(FeatureBinningTest, MaximumAndMinimumValueAreCorrectlyIdentified) {

    EXPECT_FLOAT_EQ( calculatedBinning->GetMin(), 1.0f);
    EXPECT_FLOAT_EQ( calculatedBinning->GetMax(), 12.0f);
    EXPECT_FLOAT_EQ( predefinedBinning->GetMin(), 1.0f);
    EXPECT_FLOAT_EQ( predefinedBinning->GetMax(), 12.0f);

}

TEST_F(FeatureBinningTest, NumberOfLevelsAndBinsIsCorrectlyIdentified) {

    EXPECT_EQ( calculatedBinning->GetNLevels(), 2u );
    EXPECT_EQ( predefinedBinning->GetNLevels(), 2u );
    // 5 bins, 2^2 ordinary bins + 1 NaN bin
    EXPECT_EQ( calculatedBinning->GetNBins(), 5u );
    EXPECT_EQ( predefinedBinning->GetNBins(), 5u );

}

TEST_F(FeatureBinningTest, BinRoundTrip) {

    std::vector<float> binning = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }; 
    FeatureBinning<float> featureBinning(2, binning);
    std::vector<float> values = {-1.0f, -0.5f, 0.0f, 0.1f, 0.25f, 0.3f, 0.5f, 0.6f, 0.75f, 0.9f, 1.0f, 1.2f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), NAN, std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    for(auto &x : values) {
      EXPECT_FLOAT_EQ(featureBinning.ValueToBin(x), featureBinning.ValueToBin(featureBinning.BinToValue(featureBinning.ValueToBin(x))));
    }
      
}
      
TEST_F(FeatureBinningTest, BinToValueExtensive4Layer ) {

    std::vector<float> test_binning = { 0.0f, 50.0f, 25.0f, 75.0f, 12.5f, 37.5f, 62.5f, 87.5f, 6.25f, 18.75f, 31.25f, 43.75, 56.25f, 68.75f, 81.25, 93.75, 100.0f  }; 
    std::vector<float> test_inverse_binning = { 0.0f, 6.25f, 12.5f, 18.75f, 25.0f, 31.25f, 37.5f, 43.75, 50.0f, 56.25f, 62.5f, 68.75f, 75.0f, 81.25, 87.5f, 93.75, 100.0f  }; 
    FeatureBinning<float> featureBinning(4, test_binning);

    EXPECT_TRUE(std::isnan(featureBinning.BinToValue(0u)));
    EXPECT_TRUE(std::isinf(featureBinning.BinToValue(1u)));
    EXPECT_TRUE(featureBinning.BinToValue(1u) < 0.0f);
    for(unsigned int i = 1; i < test_inverse_binning.size() - 1; ++i) {
        EXPECT_EQ(featureBinning.BinToValue(i + 1u), test_inverse_binning[i]);
    }

}

TEST_F(FeatureBinningTest, ValueToBinMapsNormalValuesCorrectly) {

    EXPECT_EQ( calculatedBinning->ValueToBin(1.0f), 1u);
    EXPECT_EQ( calculatedBinning->ValueToBin(2.0f), 1u);
    EXPECT_EQ( calculatedBinning->ValueToBin(3.0f), 1u);
    EXPECT_EQ( calculatedBinning->ValueToBin(4.0f), 2u);
    EXPECT_EQ( calculatedBinning->ValueToBin(5.0f), 2u);
    EXPECT_EQ( calculatedBinning->ValueToBin(6.0f), 2u);
    EXPECT_EQ( calculatedBinning->ValueToBin(7.0f), 3u);
    EXPECT_EQ( calculatedBinning->ValueToBin(8.0f), 3u);
    EXPECT_EQ( calculatedBinning->ValueToBin(9.0f), 3u);
    EXPECT_EQ( calculatedBinning->ValueToBin(10.0f), 4u);
    EXPECT_EQ( calculatedBinning->ValueToBin(11.0f), 4u);
    EXPECT_EQ( calculatedBinning->ValueToBin(12.0f), 4u);
    
    EXPECT_EQ( predefinedBinning->ValueToBin(1.0f), 1u);
    EXPECT_EQ( predefinedBinning->ValueToBin(2.0f), 1u);
    EXPECT_EQ( predefinedBinning->ValueToBin(3.0f), 1u);
    EXPECT_EQ( predefinedBinning->ValueToBin(4.0f), 2u);
    EXPECT_EQ( predefinedBinning->ValueToBin(5.0f), 2u);
    EXPECT_EQ( predefinedBinning->ValueToBin(6.0f), 2u);
    EXPECT_EQ( predefinedBinning->ValueToBin(7.0f), 3u);
    EXPECT_EQ( predefinedBinning->ValueToBin(8.0f), 3u);
    EXPECT_EQ( predefinedBinning->ValueToBin(9.0f), 3u);
    EXPECT_EQ( predefinedBinning->ValueToBin(10.0f), 4u);
    EXPECT_EQ( predefinedBinning->ValueToBin(11.0f), 4u);
    EXPECT_EQ( predefinedBinning->ValueToBin(12.0f), 4u);

}

TEST_F(FeatureBinningTest, BinToValueMapsNormalValuesCorrectly) {
    
    EXPECT_TRUE( std::isnan(calculatedBinning->BinToValue(0u)));
    EXPECT_EQ( calculatedBinning->BinToValue(1u), -std::numeric_limits<float>::infinity());
    EXPECT_EQ( calculatedBinning->BinToValue(2u), 4.0f);
    EXPECT_EQ( calculatedBinning->BinToValue(3u), 7.0f);
    EXPECT_EQ( calculatedBinning->BinToValue(4u), 10.0f);


    EXPECT_TRUE( std::isnan(predefinedBinning->BinToValue(0u)));
    EXPECT_EQ( predefinedBinning->BinToValue(1u), -std::numeric_limits<float>::infinity());
    EXPECT_EQ( predefinedBinning->BinToValue(2u), 4.0f);
    EXPECT_EQ( predefinedBinning->BinToValue(3u), 7.0f);
    EXPECT_EQ( predefinedBinning->BinToValue(4u), 10.0f);

}

TEST_F(FeatureBinningTest, NaNGivesZeroBin) {

    EXPECT_EQ( predefinedBinning->ValueToBin(NAN), 0u);
    EXPECT_EQ( predefinedBinning->ValueToBin(NAN), 0u);

}

TEST_F(FeatureBinningTest, OverflowAndUnderflowGivesLastAndFirstBin) {

    EXPECT_EQ( calculatedBinning->ValueToBin(100.0f), 4u);
    EXPECT_EQ( calculatedBinning->ValueToBin(-100.0f), 1u);
    EXPECT_EQ( predefinedBinning->ValueToBin(100.0f), 4u);
    EXPECT_EQ( predefinedBinning->ValueToBin(-100.0f), 1u);

}

TEST_F(FeatureBinningTest, UsingMaximumOfDoubleIsSafe) {

    EXPECT_EQ( calculatedBinning->ValueToBin(std::numeric_limits<float>::max()), 4u);
    EXPECT_EQ( calculatedBinning->ValueToBin(std::numeric_limits<float>::lowest()), 1u);
    EXPECT_EQ( predefinedBinning->ValueToBin(std::numeric_limits<float>::max()), 4u);
    EXPECT_EQ( predefinedBinning->ValueToBin(std::numeric_limits<float>::lowest()), 1u);

}


TEST_F(FeatureBinningTest, UsingInfinityIsSafe) {

    EXPECT_EQ( calculatedBinning->ValueToBin(std::numeric_limits<float>::infinity()), 4u);
    EXPECT_EQ( calculatedBinning->ValueToBin(-std::numeric_limits<float>::infinity()), 1u);
    EXPECT_EQ( predefinedBinning->ValueToBin(std::numeric_limits<float>::infinity()), 4u);
    EXPECT_EQ( predefinedBinning->ValueToBin(-std::numeric_limits<float>::infinity()), 1u);

}

TEST_F(FeatureBinningTest, GetBinningIsCorrect) {

    EXPECT_EQ( calculatedBinning->GetBinning(), binning);
    EXPECT_EQ( predefinedBinning->GetBinning(), binning);

}

TEST_F(FeatureBinningTest, ConstantFeatureIsHandledCorrectly) {

    std::vector<float> data = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }; 
    FeatureBinning<float> featureBinning(3, data);

    std::vector<float> binning = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }; 
    EXPECT_EQ( featureBinning.GetNBins(), 9u);
    EXPECT_EQ( featureBinning.GetBinning(), binning);
    EXPECT_EQ( featureBinning.ValueToBin(100.0f), 8u);
    EXPECT_EQ( featureBinning.ValueToBin(-100.0f), 1u);
    EXPECT_EQ( featureBinning.ValueToBin(1.0f), 8u);

}

TEST_F(FeatureBinningTest, FewDistinctValuesIsHandledCorrectly) {

    std::vector<float> data = { 1.0f, 1.0f, 7.0, 6.0, 1.0f, 3.0f, 3.0f, 5.0f, 2.0f, 2.0f, 2.0f, 4.0f, 1.0f, 1.0f }; 
    FeatureBinning<float> featureBinning(3, data);

    std::vector<float> binning = { 1.0f, 4.0f, 2.0f, 6.0f, 1.0, 3.0, 5.0, 7.0, 7.0f }; 
    EXPECT_EQ( featureBinning.GetNBins(), 9u);
    EXPECT_EQ( featureBinning.GetBinning(), binning);
    EXPECT_EQ( featureBinning.ValueToBin(100.0f), 8u);
    EXPECT_EQ( featureBinning.ValueToBin(-100.0f), 1u);
    EXPECT_EQ( featureBinning.ValueToBin(1.0f), 2u);
    EXPECT_EQ( featureBinning.ValueToBin(2.0f), 3u);
    EXPECT_EQ( featureBinning.ValueToBin(3.0f), 4u);
    EXPECT_EQ( featureBinning.ValueToBin(4.0f), 5u);
    EXPECT_EQ( featureBinning.ValueToBin(5.0f), 6u);
    EXPECT_EQ( featureBinning.ValueToBin(6.0f), 7u);
    EXPECT_EQ( featureBinning.ValueToBin(7.0f), 8u);

}

TEST_F(FeatureBinningTest, LowStatisticIsHandledCorrectly) {

    std::vector<float> data = { 1.0f, 4.0f, 4.0f, 7.0f, 10.0f, 11.0f, 12.0f }; 
    FeatureBinning<float> featureBinning(3, data);

    std::vector<float> binning = { 1.0f, 10.0f, 4.0f, 12.0f, 1.0f, 7.0f, 11.0f, 12.0f, 12.0f }; 
    EXPECT_EQ( featureBinning.GetNBins(), 9u);
    EXPECT_EQ( featureBinning.GetBinning(), binning);
    
    EXPECT_EQ( featureBinning.ValueToBin(100.0f), 8u);
    EXPECT_EQ( featureBinning.ValueToBin(-100.0f), 1u);
    
    EXPECT_EQ( featureBinning.ValueToBin(1.0f), 2u);
    EXPECT_EQ( featureBinning.ValueToBin(2.0f), 2u);
    EXPECT_EQ( featureBinning.ValueToBin(3.0f), 2u);
    EXPECT_EQ( featureBinning.ValueToBin(4.0f), 3u);
    EXPECT_EQ( featureBinning.ValueToBin(5.0f), 3u);
    EXPECT_EQ( featureBinning.ValueToBin(6.0f), 3u);
    EXPECT_EQ( featureBinning.ValueToBin(7.0f), 4u);
    EXPECT_EQ( featureBinning.ValueToBin(8.0f), 4u);
    EXPECT_EQ( featureBinning.ValueToBin(9.0f), 4u);
    EXPECT_EQ( featureBinning.ValueToBin(10.0f), 5u);
    EXPECT_EQ( featureBinning.ValueToBin(11.0f), 6u);
    EXPECT_EQ( featureBinning.ValueToBin(12.0f), 8u);
    
    FeatureBinning<float> featureBinning2(4, data);

    std::vector<float> binning2 = { 1.0f, 12.0f, 10.0f, 12.0f, 4.0f, 12.0f, 12.0f, 12.0f, 1.0f, 7.0f, 11.0f, 12.0f, 12.0f, 12.0f, 12.0f, 12.0f, 12.0f }; 
    EXPECT_EQ( featureBinning2.GetNBins(), 17u);
    EXPECT_EQ( featureBinning2.GetBinning(), binning2);
    
    EXPECT_EQ( featureBinning2.ValueToBin(100.0f), 16u);
    EXPECT_EQ( featureBinning2.ValueToBin(-100.0f), 1u);
    
    EXPECT_EQ( featureBinning2.ValueToBin(1.0f), 2u);
    EXPECT_EQ( featureBinning2.ValueToBin(2.0f), 2u);
    EXPECT_EQ( featureBinning2.ValueToBin(3.0f), 2u);
    EXPECT_EQ( featureBinning2.ValueToBin(4.0f), 3u);
    EXPECT_EQ( featureBinning2.ValueToBin(5.0f), 3u);
    EXPECT_EQ( featureBinning2.ValueToBin(6.0f), 3u);
    EXPECT_EQ( featureBinning2.ValueToBin(7.0f), 4u);
    EXPECT_EQ( featureBinning2.ValueToBin(8.0f), 4u);
    EXPECT_EQ( featureBinning2.ValueToBin(9.0f), 4u);
    EXPECT_EQ( featureBinning2.ValueToBin(10.0f), 5u);
    EXPECT_EQ( featureBinning2.ValueToBin(11.0f), 6u);
    EXPECT_EQ( featureBinning2.ValueToBin(12.0f), 16u);
    
}

class WeightedFeatureBinningTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            std::vector<float> data = {10.0f,8.0f,2.0f,NAN,NAN,NAN,NAN,7.0f,5.0f,6.0f,9.0f,NAN,4.0f,3.0f,11.0f,12.0f,1.0f,NAN};
            std::vector<Weight> weights = {2.0f,0.1f,0.1f,3.0f,0.5f,1.0f,2.0f,0.1f,2.0f,3.0f,1.0f,0.1f,1.0f,2.0f,2.0f,1.0f,0.5f,12.0f};
            calculatedBinning = new WeightedFeatureBinning<float>(2, data, weights);

            binning = { 1.0f, 6.0f, 5.0f, 10.0f, 12.0f }; 
            predefinedBinning = new FeatureBinning<float>(2, binning);
            
            // Set the binning again, because it is sorted inside the constructor
            binning = { 1.0f, 6.0f, 5.0f, 10.0f, 12.0f }; 
        }

        virtual void TearDown() {
            delete calculatedBinning;
            delete predefinedBinning;
        }

        unsigned int nLevels;
        std::vector<float> binning;
        WeightedFeatureBinning<float> *calculatedBinning;
        FeatureBinning<float> *predefinedBinning;

};

TEST_F(WeightedFeatureBinningTest, MaximumAndMinimumValueAreCorrectlyIdentified) {

    EXPECT_FLOAT_EQ( calculatedBinning->GetMin(), 1.0f);
    EXPECT_FLOAT_EQ( calculatedBinning->GetMax(), 12.0f);
    EXPECT_FLOAT_EQ( predefinedBinning->GetMin(), 1.0f);
    EXPECT_FLOAT_EQ( predefinedBinning->GetMax(), 12.0f);

}

TEST_F(WeightedFeatureBinningTest, NumberOfLevelsAndBinsIsCorrectlyIdentified) {

    EXPECT_EQ( calculatedBinning->GetNLevels(), 2u );
    EXPECT_EQ( predefinedBinning->GetNLevels(), 2u );
    // 5 bins, 2^2 ordinary bins + 1 NaN bin
    EXPECT_EQ( calculatedBinning->GetNBins(), 5u );
    EXPECT_EQ( predefinedBinning->GetNBins(), 5u );

}

TEST_F(WeightedFeatureBinningTest, GetBinningIsCorrect) {

    EXPECT_EQ( calculatedBinning->GetBinning(), binning);
    EXPECT_EQ( predefinedBinning->GetBinning(), binning);
}

TEST_F(WeightedFeatureBinningTest, SameAsUsualBinningWithoutWeights) {
    std::vector<float> data = {10.0f,8.0f,2.0f,NAN,NAN,NAN,NAN,7.0f,5.0f,6.0f,9.0f,NAN,4.0f,3.0f,11.0f,12.0f,1.0f,NAN};
    std::vector<Weight> weights(data.size(), 1.0);
    FeatureBinning<float> usualBinning(2, data);
    WeightedFeatureBinning<float> weightedBinning(2, data, weights);
    
    EXPECT_EQ( usualBinning.GetBinning(), weightedBinning.GetBinning() );

}

class EquidistantFeatureBinningTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            std::vector<float> data = {10.0f,8.0f,8.0, 8.0,16.0,2.0f,NAN,NAN,NAN,NAN,7.0f,4.0f,4.0f,9.0f,NAN,4.0f,3.0f,1.0f,0.0f,1.0f,NAN};
            calculatedBinning = new EquidistantFeatureBinning<float>(2, data);

            binning = { 0.0, 8.0f, 4.0f, 12.0f, 16.0f }; 
        }

        virtual void TearDown() {
            delete calculatedBinning;
        }

        unsigned int nLevels;
        std::vector<float> binning;
        EquidistantFeatureBinning<float> *calculatedBinning;

};

TEST_F(EquidistantFeatureBinningTest, MaximumAndMinimumValueAreCorrectlyIdentified) {

    EXPECT_FLOAT_EQ( calculatedBinning->GetMin(), 0.0f);
    EXPECT_FLOAT_EQ( calculatedBinning->GetMax(), 16.0f);

}

TEST_F(EquidistantFeatureBinningTest, GetBinningIsCorrect) {

    EXPECT_EQ( calculatedBinning->GetBinning(), binning);
}


class PurityTransformationTest : public ::testing::Test {
    protected:
        virtual void SetUp() {

            binned_data = {0, 0, 0, 0, 0,
                           1, 1, 1, 1,
                           2, 2, 2, 2, 2,
                           3, 3, 3,
                           4, 4};
            weights = {1.0, 2.0, 3.0, 4.0, 5.0,
                       1.0, 2.0, 3.0, 4.0,
                       1.0, 2.0, 3.0, 4.0, 5.0,
                       1.0, 2.0, 3.0,
                       1.0, 2.0};
            isSignal = {true, false, true, false, true, // 9 / 15
                        false, false, true, true, // 7 / 10
                        true, true, true, true, false, // 10 / 15
                        true, true, false, // 3 / 6
                        true, false, // 1 / 3
            };

        }

        virtual void TearDown() {
        }

        std::vector<unsigned int> binned_data;
        std::vector<float> weights;
        std::vector<bool> isSignal;

};

TEST_F(PurityTransformationTest, TestPurityTransformation) {

    PurityTransformation pt(2, binned_data, weights, isSignal);

    EXPECT_EQ(pt.BinToPurityBin(0), 0u);
    EXPECT_EQ(pt.BinToPurityBin(1), 4u);
    EXPECT_EQ(pt.BinToPurityBin(2), 3u);
    EXPECT_EQ(pt.BinToPurityBin(3), 2u);
    EXPECT_EQ(pt.BinToPurityBin(4), 1u);

}

class EventWeightsTest : public ::testing::Test {

    protected:
        virtual void SetUp() {
            eventWeights = new EventWeights(10);
            for(unsigned int i = 0; i < 10; ++i) {
                eventWeights->Set(i, static_cast<Weight>(i+1));
                eventWeights->SetOriginal(i, 2);
            }
        }

        virtual void TearDown() {
            delete eventWeights;
        }

        EventWeights *eventWeights;
};

TEST_F(EventWeightsTest, WeightSumsAreCorrect) {

    auto sums = eventWeights->GetSums(5);
    EXPECT_FLOAT_EQ(sums[0], 15.0 * 2);
    EXPECT_FLOAT_EQ(sums[1], 40.0 * 2);
    EXPECT_FLOAT_EQ(sums[2], 385.0 * 2);

}

TEST_F(EventWeightsTest, WeightSumsAreNotInfluencedByZeroWeights) {

    auto sums = eventWeights->GetSums(5);
            
    EventWeights *newEventWeights = new EventWeights(20);
    for(unsigned int i = 0; i < 10; ++i) {
        // Get delivers the weight*original weight, therefore we need to divide by the original weight afterwards
        newEventWeights->Set(i*2, eventWeights->Get(i) / eventWeights->GetOriginal(i));
        newEventWeights->SetOriginal(i*2, eventWeights->GetOriginal(i));
        newEventWeights->Set(i*2 + 1, 0.0);
        newEventWeights->SetOriginal(i*2 + 1, 0.0);
    }
    auto newSums = newEventWeights->GetSums(10);
    delete newEventWeights;

    EXPECT_FLOAT_EQ(sums[0], newSums[0]);
    EXPECT_FLOAT_EQ(sums[1], newSums[1]);
    EXPECT_FLOAT_EQ(sums[2], newSums[2]);

}

TEST_F(EventWeightsTest, GetterIsCorrect) {

    for(unsigned int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ( eventWeights->Get(i), static_cast<Weight>(i+1) * 2); 
    }
    
}

TEST_F(EventWeightsTest, WeightSumsAndGetterAreCorrectlyUpdated) {

    for(unsigned int i = 0; i < 10; ++i) {
        eventWeights->Set(i, static_cast<Weight>(i+3));
    }

    auto sums = eventWeights->GetSums(5);
    EXPECT_FLOAT_EQ(sums[0], 25.0 * 2);
    EXPECT_FLOAT_EQ(sums[1], 50.0 * 2);
    EXPECT_FLOAT_EQ(sums[2], 645.0 * 2);
    
    for(unsigned int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ( eventWeights->Get(i), static_cast<Weight>(i+3) * 2); 
    }

}

class EventFlagsTest : public ::testing::Test {

    protected:
        virtual void SetUp() {
            eventFlags = new EventFlags(10);
        }

        virtual void TearDown() {
            delete eventFlags;
        }

        EventFlags *eventFlags;
};

TEST_F(EventFlagsTest, IsInitialisedWithOnes) {

    for(unsigned int i = 0; i < 10; ++i)
        EXPECT_EQ( eventFlags->Get(i), 1);

}

TEST_F(EventFlagsTest, SetterAndGetterWorkCorrectly) {
    
    for(unsigned int i = 0; i < 10; ++i)
        eventFlags->Set(i, i-5 );

    for(unsigned int i = 0; i < 10; ++i)
        EXPECT_EQ( eventFlags->Get(i), static_cast<int>(i)-5);

}

class EventValuesTest : public ::testing::Test {

    protected:
        virtual void SetUp() {
            eventValues = new EventValues(8, 4, {3, 4, 2, 3});
        }

        virtual void TearDown() {
            delete eventValues;
        }

        EventValues *eventValues;
};

TEST_F(EventValuesTest, SetterAndGetterWorkCorrectly) {
    
    for(unsigned int i = 0; i < 8; ++i) {
        std::vector<unsigned int> features = { i, static_cast<unsigned int>(4 + (1-2*((int)i%2))*((int)i+1)/2), static_cast<unsigned int>((int)(i) % 4 + 1),  7-i };
        eventValues->Set(i, features);
    }
    EXPECT_THROW( eventValues->Set(1, {1,2,3,4,5}), std::runtime_error );
    EXPECT_THROW( eventValues->Set(1, {1,20,3,1}), std::runtime_error );
    
    for(unsigned int i = 0; i < 8; ++i) {
        std::vector<unsigned int> features = { i, static_cast<unsigned int>(4 + (1-2*((int)(i)%2))*((int)(i)+1)/2), static_cast<unsigned int>((int)(i) % 4 + 1),  7-i };
        const auto *array = &eventValues->Get(i);
        for(unsigned int j = 0; j < 3; ++j) {
            EXPECT_EQ( eventValues->Get(i,j), features[j]);
            EXPECT_EQ( array[j], features[j]);
        }
    }
}


TEST_F(EventValuesTest, ThrowOnMismatchBetweenNFeaturesAndNBinsSize) {
    
  EXPECT_THROW( EventValues(8, 3, {1, 2}), std::runtime_error );

}


TEST_F(EventValuesTest, GetSizesWorkCorrectly) {

    EXPECT_EQ( eventValues->GetNFeatures(), 4u);
    const auto& nBins = eventValues->GetNBins();
    EXPECT_EQ( nBins.size(), 4u);
    EXPECT_EQ( nBins[0], 9u);
    EXPECT_EQ( nBins[1], 17u);
    EXPECT_EQ( nBins[2], 5u);
    EXPECT_EQ( nBins[3], 9u);
    
    const auto& nBinSums = eventValues->GetNBinSums();
    EXPECT_EQ( nBinSums.size(), 5u);
    EXPECT_EQ( nBinSums[0], 0u);
    EXPECT_EQ( nBinSums[1], 9u);
    EXPECT_EQ( nBinSums[2], 9u + 17u);
    EXPECT_EQ( nBinSums[3], 9u + 17u + 5u);
    EXPECT_EQ( nBinSums[4], 9u + 17u + 5u + 9u);

}

class EventSampleTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(10, 3, {8, 8, 8});
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;

};

TEST_F(EventSampleTest, AddingEventsWorksCorrectly) {

    eventSample->AddEvent( std::vector<unsigned int>({1,2,3}), 2.0, true );
    EXPECT_EQ( eventSample->GetNSignals(), 1u);
    EXPECT_EQ( eventSample->GetNBckgrds(), 0u);
   
    const auto &eventWeights = eventSample->GetWeights();
    auto sums = eventWeights.GetSums(5);
    EXPECT_FLOAT_EQ(sums[0], 2.0);
    EXPECT_FLOAT_EQ(sums[1], 0.0);

 
    // Add some more Signal and Background events   
    for(unsigned int i = 1; i < 10; ++i) { 
        eventSample->AddEvent( std::vector<unsigned int>({2*i,3*i,5*i}), 2.0, i % 2 == 0 );
    }
    EXPECT_EQ( eventSample->GetNSignals(), 5u);
    EXPECT_EQ( eventSample->GetNBckgrds(), 5u);
    
    sums = eventWeights.GetSums(5);
    EXPECT_FLOAT_EQ(sums[0], 10.0);
    EXPECT_FLOAT_EQ(sums[1], 10.0);
    
    // Test some of the values, if they're correct
    // Remember that the events are NOT in the same order as they were added,
    // instead the signal events are added starting from 0, and the background events
    // are added reversed starting from the last event.
    EXPECT_EQ( eventSample->GetValues().Get(1,2), 10u); 
    EXPECT_EQ( eventSample->GetValues().Get(3,1), 18u); 
    EXPECT_EQ( eventSample->GetValues().Get(9,0), 2u); 

    // Test if signal and background labels are correctly assigned
    for(unsigned int i = 0; i < 5; ++i) {
        EXPECT_TRUE( eventSample->IsSignal(i));
        EXPECT_FALSE( eventSample->IsSignal(i+5));
    }

    // Test throw if number of promised events is exceeded
    EXPECT_THROW( eventSample->AddEvent( std::vector<unsigned int>({1,2,3}), 2.0, true ), std::runtime_error);
    
}

TEST_F(EventSampleTest, AddingEventsWithZeroWeightWorksCorrectly) {
 
    // Add some more Signal and Background events   
    for(unsigned int i = 0; i < 10; ++i) { 
        eventSample->AddEvent( std::vector<unsigned int>({2*i,3*i,5*i}), i % 3, i % 2 == 0 );
    }
    EXPECT_EQ( eventSample->GetNSignals(), 5u);
    EXPECT_EQ( eventSample->GetNBckgrds(), 5u);
    
    const auto &eventWeights = eventSample->GetWeights();
    const auto& sums = eventWeights.GetSums(5);
    EXPECT_FLOAT_EQ(sums[0], 5.0);
    EXPECT_FLOAT_EQ(sums[1], 4.0);

}

TEST_F(EventSampleTest, AddingEventsWithNANWeightsTrhow) {
 
    eventSample->AddEvent( std::vector<unsigned int>({2,3,5}), 1.0, true);
    EXPECT_THROW(eventSample->AddEvent( std::vector<unsigned int>({2,3,5}), NAN, true), std::runtime_error);

}

class CumulativeDistributionsTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            const unsigned int numberOfEvents = 100;
            eventSample = new EventSample(numberOfEvents, 2, {2, 2});
            for(unsigned int i = 0; i < numberOfEvents; ++i) {
                bool isSignal = i < (numberOfEvents/2);
                eventSample->AddEvent( std::vector<unsigned int>({i % 4 + 1, (numberOfEvents-i) % 4 + 1}), static_cast<Weight>(i+1), isSignal);
            }
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;
};

TEST_F(CumulativeDistributionsTest, CheckIfLayer0IsCorrect) {

    CumulativeDistributions CDFsForLayer0(0, *eventSample);

    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 1), 325.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 2), 663.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 3), 963.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 4), 1275.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 1), 325.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 2), 637.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 3), 937.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 4), 1275.0); 

    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 1), 900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 2), 1812.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 3), 2787.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 4), 3775.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 1), 900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 2), 1888.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 3), 2863.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 4), 3775.0); 

}

TEST_F(CumulativeDistributionsTest, NaNShouldBeIgnored) {

    CumulativeDistributions CDFsForLayer0(0, *eventSample);
            
    std::vector<unsigned int> v(2);
    EventSample *newEventSample = new EventSample(200, 2, {2, 2});
    for(unsigned int i = 0; i < 100; ++i) {
        v[0] = eventSample->GetValues().Get(i, 0);
        v[1] = eventSample->GetValues().Get(i, 1);
        newEventSample->AddEvent(v, eventSample->GetWeights().GetOriginal(i), eventSample->IsSignal(i));
        newEventSample->AddEvent(std::vector<unsigned int>({0, 0}), 1.0, i < 50);
    }
    CumulativeDistributions newCDFsForLayer0(0, *newEventSample);
    delete newEventSample;

    for(unsigned int iBin = 1; iBin < 5; ++iBin) {
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, iBin), newCDFsForLayer0.GetSignal(0, 0, iBin)); 
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, iBin), newCDFsForLayer0.GetBckgrd(0, 0, iBin)); 
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, iBin), newCDFsForLayer0.GetSignal(0, 1, iBin)); 
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, iBin), newCDFsForLayer0.GetBckgrd(0, 1, iBin)); 
    }

    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 0), 0.0);
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 0), 0.0);
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 0), 0.0);
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 0), 0.0);
    
    EXPECT_FLOAT_EQ( newCDFsForLayer0.GetSignal(0, 0, 0), 50.0);
    EXPECT_FLOAT_EQ( newCDFsForLayer0.GetBckgrd(0, 0, 0), 50.0);
    EXPECT_FLOAT_EQ( newCDFsForLayer0.GetSignal(0, 1, 0), 50.0);
    EXPECT_FLOAT_EQ( newCDFsForLayer0.GetBckgrd(0, 1, 0), 50.0);

}

TEST_F(CumulativeDistributionsTest, ZeroWeightShouldBeIgnored) {

    CumulativeDistributions CDFsForLayer0(0, *eventSample);
            
    std::vector<unsigned int> v(2);
    EventSample *newEventSample = new EventSample(200, 2, {2, 2});
    for(unsigned int i = 0; i < 100; ++i) {
        v[0] = eventSample->GetValues().Get(i, 0);
        v[1] = eventSample->GetValues().Get(i, 1);
        newEventSample->AddEvent(v, eventSample->GetWeights().GetOriginal(i), eventSample->IsSignal(i));
        newEventSample->AddEvent(std::vector<unsigned int>({i%2 + 1, i%3 + 1}), 0.0, i < 50);
    }
    CumulativeDistributions newCDFsForLayer0(0, *newEventSample);
    delete newEventSample;

    for(unsigned int iBin = 0; iBin < 5; ++iBin) {
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, iBin), newCDFsForLayer0.GetSignal(0, 0, iBin)); 
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, iBin), newCDFsForLayer0.GetBckgrd(0, 0, iBin)); 
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, iBin), newCDFsForLayer0.GetSignal(0, 1, iBin)); 
      EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, iBin), newCDFsForLayer0.GetBckgrd(0, 1, iBin)); 
    }

}


TEST_F(CumulativeDistributionsTest, CheckIfLayer1IsCorrect) {
    
    auto &eventFlags = eventSample->GetFlags();
    for(unsigned int i = 0; i < 50; ++i) {
        eventFlags.Set(i, i%2 + 2 );
    }
    for(unsigned int i = 50; i < 100; ++i) {
        eventFlags.Set(149-i, i%2 + 2 );
    }

    CumulativeDistributions CDFsForLayer1(1, *eventSample);

    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 0, 1), 325.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 0, 2), 325.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 0, 3), 625.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 0, 4), 625.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 1, 1), 325.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 1, 2), 325.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 1, 3), 625.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 1, 4), 625.0); 

    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 0, 1), 900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 0, 2), 900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 0, 3), 1875.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 0, 4), 1875.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 1, 1), 900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 1, 2), 900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 1, 3), 1875.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 1, 4), 1875.0); 

    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 0, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 0, 2), 338.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 0, 3), 338.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 0, 4), 650.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 1, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 1, 2), 312.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 1, 3), 312.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 1, 4), 650.0); 

    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 0, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 0, 2), 912.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 0, 3), 912.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 0, 4), 1900.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 1, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 1, 2), 988.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 1, 3), 988.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 1, 4), 1900.0); 

}

TEST_F(CumulativeDistributionsTest, DifferentBinningLevels) {
    const unsigned int numberOfEvents = 10;
    EventSample *sample = new EventSample(numberOfEvents, 4, {2, 1, 3, 1});
    sample->AddEvent(std::vector<unsigned int>{3, 1, 8, 2}, 1.0, true); 
    sample->AddEvent(std::vector<unsigned int>{4, 2, 7, 2}, 1.0, true); 
    sample->AddEvent(std::vector<unsigned int>{3, 2, 6, 0}, 1.0, true); 
    sample->AddEvent(std::vector<unsigned int>{2, 1, 5, 1}, 1.0, true); 
    sample->AddEvent(std::vector<unsigned int>{1, 1, 4, 1}, 1.0, true); 
    sample->AddEvent(std::vector<unsigned int>{3, 1, 3, 2}, 1.0, false); 
    sample->AddEvent(std::vector<unsigned int>{4, 2, 2, 2}, 1.0, false); 
    sample->AddEvent(std::vector<unsigned int>{3, 2, 1, 0}, 1.0, false); 
    sample->AddEvent(std::vector<unsigned int>{2, 1, 2, 1}, 1.0, false); 
    sample->AddEvent(std::vector<unsigned int>{1, 1, 3, 2}, 1.0, false); 
    
    CumulativeDistributions CDFsForLayer0(0, *sample);

    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 1), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 2), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 3), 4.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 0, 4), 5.0);

    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 1), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 1, 2), 5.0); 
    
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 2), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 3), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 4), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 5), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 6), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 7), 4.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 2, 8), 5.0);  

    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 3, 0), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 3, 1), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetSignal(0, 3, 2), 4.0); 
    
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 1), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 2), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 3), 4.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 0, 4), 5.0); 
    
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 1), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 1, 2), 5.0); 
    
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 1), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 2), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 3), 5.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 4), 5.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 5), 5.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 6), 5.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 7), 5.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 2, 8), 5.0); 
    
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 3, 0), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 3, 1), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer0.GetBckgrd(0, 3, 2), 4.0); 
    
    auto &eventFlags = sample->GetFlags();
    for(unsigned int i = 0; i < 10; ++i) {
      eventFlags.Set(i, i%2 + 2);
    }

    // We check only the third feature here, if something goes wrong
    // due to the different binning sizes this should influence this feature.
    CumulativeDistributions CDFsForLayer1(1, *sample);
    
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 2), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 3), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 4), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 5), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 6), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 7), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(0, 2, 8), 3.0);  
    
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 2), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 3), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 4), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 5), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 6), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 7), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(0, 2, 8), 2.0); 
    
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 1), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 2), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 3), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 4), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 5), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 6), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 7), 2.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetSignal(1, 2, 8), 2.0);  
    
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 0), 0.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 1), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 2), 1.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 3), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 4), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 5), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 6), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 7), 3.0); 
    EXPECT_FLOAT_EQ( CDFsForLayer1.GetBckgrd(1, 2, 8), 3.0); 

    delete sample;
}

class LossFunctionTest : public ::testing::Test { };

TEST_F(LossFunctionTest, GiniIndexIsCorrect) {

    EXPECT_FLOAT_EQ( LossFunction(4,4), 2.0);
    EXPECT_FLOAT_EQ( LossFunction(1,4), 0.8);
    EXPECT_FLOAT_EQ( LossFunction(4,1), 0.8);
    EXPECT_FLOAT_EQ( LossFunction(2,0), 0.0);
    EXPECT_FLOAT_EQ( LossFunction(0,2), 0.0);

}

class NodeTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(8, 2, {1, 1});
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 4.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 4.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 3.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 2.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 3.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 2.0, false);
            
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;
};

TEST_F(NodeTest, IsInLayerIsCorrect) {

    EXPECT_TRUE( Node(0,0).IsInLayer(0) );
    EXPECT_FALSE( Node(0,0).IsInLayer(1) );
    EXPECT_TRUE( Node(1,0).IsInLayer(1) );
    EXPECT_FALSE( Node(1,0).IsInLayer(0) );
    EXPECT_FALSE( Node(1,0).IsInLayer(2) );

}

TEST_F(NodeTest, PositionIsCorrectlyDetermined) {

    EXPECT_EQ( Node(0,0).GetPosition(), 0u );
    EXPECT_EQ( Node(1,0).GetPosition(), 1u );
    EXPECT_EQ( Node(1,1).GetPosition(), 2u );
    EXPECT_EQ( Node(2,0).GetPosition(), 3u );
    EXPECT_EQ( Node(2,1).GetPosition(), 4u );
    EXPECT_EQ( Node(2,2).GetPosition(), 5u );
    EXPECT_EQ( Node(2,3).GetPosition(), 6u );
    EXPECT_EQ( Node(3,0).GetPosition(), 7u );
    EXPECT_EQ( Node(3,7).GetPosition(), 14u );

}

TEST_F(NodeTest, BoostWeightCalculation) {

    Node node(0,0);
    node.SetWeights({2.0, 2.0, 4.0});
    EXPECT_FLOAT_EQ(node.GetBoostWeight(), 0.0); 
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(1.0, 1.0);
    node.AddSignalWeight(3.0, 1.0);
    node.AddBckgrdWeight(2.0, 1.0);
    EXPECT_FLOAT_EQ(node.GetBoostWeight(), -1.0);

}

TEST_F(NodeTest, PurityCalculation) {

    Node node(0,0);
    node.SetWeights({2.0, 2.0, 4.0});
    EXPECT_FLOAT_EQ(node.GetPurity(), 0.5); 
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(2.0, 1.0);
    node.AddSignalWeight(4.0, 1.0);
    node.AddBckgrdWeight(4.0, 1.0);
    EXPECT_FLOAT_EQ(node.GetPurity(), 0.6);

}

TEST_F(NodeTest, NegativeWeightsAreHandledCorrectly) {

    Node node(0,0);
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(-2.0, -1.0);
    node.AddSignalWeight(-4.0, -1.0);
    node.AddBckgrdWeight(-4.0, -1.0);
    EXPECT_FLOAT_EQ(node.GetPurity(), 0.6);
    EXPECT_FLOAT_EQ(node.GetBoostWeight(), -0.125);
    
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(-2.0, 1.0);
    node.AddSignalWeight(1.0, -2.0);
    node.AddBckgrdWeight(0.5, -0.5);
    // Purity above 1.0 can happen with negative weights
    EXPECT_FLOAT_EQ(node.GetPurity(), 2.0);
    EXPECT_FLOAT_EQ(node.GetBoostWeight(), 0.375);

}


TEST_F(NodeTest, AddZeroWeightDoesNotChangeAnything) {

    Node node(0,0);
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(2.0, 1.0);
    node.AddSignalWeight(2.0, -1.0);
    node.AddSignalWeight(4.0, 1.0);
    node.AddSignalWeight(-4.0, 2.0);
    node.AddBckgrdWeight(4.0, 1.0);
    node.AddBckgrdWeight(4.0, 1.0);
    node.AddBckgrdWeight(3.0, -1.0);
    node.AddBckgrdWeight(2.0, 2.0);
    node.AddBckgrdWeight(0.5, 0.1);
    
    Node newNode(0,0);
    newNode.SetWeights({0.0, 0.0, 0.0});
    newNode.AddSignalWeight(2.0, 1.0);
    newNode.AddSignalWeight(2.0, -1.0);
    newNode.AddSignalWeight(2.0, 0.0);
    newNode.AddSignalWeight(4.0, 1.0);
    newNode.AddSignalWeight(-4.0, 2.0);
    newNode.AddSignalWeight(-4.0, 0.0);
    newNode.AddBckgrdWeight(4.0, 1.0);
    newNode.AddBckgrdWeight(4.0, 0.0);
    newNode.AddBckgrdWeight(4.0, 1.0);
    newNode.AddBckgrdWeight(3.0, -1.0);
    newNode.AddBckgrdWeight(2.0, 2.0);
    newNode.AddBckgrdWeight(0.0, 0.0);
    newNode.AddBckgrdWeight(0.5, 0.1);
    

    EXPECT_FLOAT_EQ(node.GetPurity(), newNode.GetPurity());
    EXPECT_FLOAT_EQ(node.GetBoostWeight(), newNode.GetBoostWeight());

}

TEST_F(NodeTest, BestCut0Layer) {

    CumulativeDistributions CDFs(0, *eventSample);
    Node node(0,0);
    node.SetWeights({10.0, 10.0, 68.0});

    auto bestCut = node.CalculateBestCut(CDFs);
    EXPECT_EQ( bestCut.feature, 0u );
    EXPECT_EQ( bestCut.index, 2u );
    EXPECT_FLOAT_EQ( bestCut.gain, 1.875 );
    EXPECT_TRUE( bestCut.valid );

}

TEST_F(NodeTest, NaNIsIgnored) {

    CumulativeDistributions CDFs(0, *eventSample);
    Node node(0,0);
    node.SetWeights({10.0, 10.0, 68.0});
    auto bestCut = node.CalculateBestCut(CDFs);
    
    EXPECT_FLOAT_EQ(CDFs.GetSignal(0, 0, 0), 0.0);
    EXPECT_FLOAT_EQ(CDFs.GetBckgrd(0, 0, 0), 0.0);
    EXPECT_FLOAT_EQ(CDFs.GetSignal(0, 1, 0), 0.0);
    EXPECT_FLOAT_EQ(CDFs.GetBckgrd(0, 1, 0), 0.0);
    // I violate constness here because it's the simplest way to test the influence
    // of the 0th bin, which contains the weights for the NaN values.
    // Signal and Background are chosen extremly asymmetric for both features, so
    // this should change the cut if the 0th bin is considered.
    const_cast<Weight&>(CDFs.GetSignal(0, 0, 0)) = 100.0;
    const_cast<Weight&>(CDFs.GetBckgrd(0, 0, 0)) = 1.0;
    const_cast<Weight&>(CDFs.GetSignal(0, 1, 0)) = 10.0;
    const_cast<Weight&>(CDFs.GetBckgrd(0, 1, 0)) = 800.0;
    auto newBestCut = node.CalculateBestCut(CDFs);

    EXPECT_EQ( bestCut.feature, newBestCut.feature );
    EXPECT_EQ( bestCut.index, newBestCut.index );
    EXPECT_FLOAT_EQ( bestCut.gain, newBestCut.gain );
    EXPECT_EQ( bestCut.valid, newBestCut.valid );

}

TEST_F(NodeTest, BestCut1Layer) {

    auto &flags = eventSample->GetFlags();
    flags.Set(0, 2);
    flags.Set(1, 2);
    flags.Set(2, 2);
    flags.Set(3, 3);
    flags.Set(4, 3);
    flags.Set(5, 2);
    flags.Set(6, 3);
    flags.Set(7, 3);

    CumulativeDistributions CDFs(1, *eventSample);

    Node right_node(1,0);
    right_node.SetWeights({7.0, 1.0, 22.0});
    auto right_bestCut = right_node.CalculateBestCut(CDFs);
    EXPECT_EQ( right_bestCut.feature, 1u );
    EXPECT_EQ( right_bestCut.index, 2u );
    EXPECT_FLOAT_EQ( right_bestCut.gain, 0.375);
    EXPECT_TRUE( right_bestCut.valid );
    
    Node left_node(1,1);
    left_node.SetWeights({3.0, 9.0, 38.0});
    auto left_bestCut = left_node.CalculateBestCut(CDFs);
    EXPECT_EQ( left_bestCut.feature, 1u );
    EXPECT_EQ( left_bestCut.index, 2u );
    EXPECT_FLOAT_EQ( left_bestCut.gain, 0.53571428571428581);
    EXPECT_TRUE( left_bestCut.valid );

}

class TreeBuilderTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(8, 2, {1, 1});
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);

            auto &weights = eventSample->GetWeights();
            weights.Set(0, 4.0);
            weights.Set(1, 1.0);
            weights.Set(2, 2.0);
            weights.Set(3, 3.0);
            weights.Set(4, 2.0);
            weights.Set(5, 1.0);
            weights.Set(6, 3.0);
            weights.Set(7, 4.0);
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;
};

TEST_F(TreeBuilderTest, DeterminedCutsAreCorrect) {
    
    TreeBuilder dt(2, *eventSample);
    const auto &cuts = dt.GetCuts();
    EXPECT_EQ( cuts[0].feature, 0u );
    EXPECT_EQ( cuts[0].index, 2u );
    EXPECT_FLOAT_EQ( cuts[0].gain, 1.875 );
    EXPECT_TRUE( cuts[0].valid );
    
    EXPECT_EQ( cuts[1].feature, 1u );
    EXPECT_EQ( cuts[1].index, 2u );
    EXPECT_FLOAT_EQ( cuts[1].gain, 0.375 );
    EXPECT_TRUE( cuts[1].valid );
    
    EXPECT_EQ( cuts[2].feature, 1u );
    EXPECT_EQ( cuts[2].index, 2u );
    EXPECT_FLOAT_EQ( cuts[2].gain, 0.53571428571428581 );
    EXPECT_TRUE( cuts[2].valid );

}

TEST_F(TreeBuilderTest, FlagsAreCorrectAfterTraining) {
    
    TreeBuilder dt(2, *eventSample);
    auto &flags = eventSample->GetFlags();
    EXPECT_EQ( flags.Get(0), 4 );
    EXPECT_EQ( flags.Get(1), 5 );
    EXPECT_EQ( flags.Get(2), 4 );
    EXPECT_EQ( flags.Get(3), 6 );
    EXPECT_EQ( flags.Get(4), 7 );
    EXPECT_EQ( flags.Get(5), 5 );
    EXPECT_EQ( flags.Get(6), 7 );
    EXPECT_EQ( flags.Get(7), 6 );

}

TEST_F(TreeBuilderTest, NEntriesOfNodesAreCorrectAfterTraining) {
    
    TreeBuilder dt(2, *eventSample);
    const auto &nEntries = dt.GetNEntries();
    EXPECT_FLOAT_EQ( nEntries[0], 20.0 );
    EXPECT_FLOAT_EQ( nEntries[1], 8.0 );
    EXPECT_FLOAT_EQ( nEntries[2], 12.0 );
    EXPECT_FLOAT_EQ( nEntries[3], 6.0 );
    EXPECT_FLOAT_EQ( nEntries[4], 2.0 );
    EXPECT_FLOAT_EQ( nEntries[5], 7.0 );
    EXPECT_FLOAT_EQ( nEntries[6], 5.0 );

}


TEST_F(TreeBuilderTest, PuritiesOfNodesAreCorrectAfterTraining) {
    
    TreeBuilder dt(2, *eventSample);
    const auto &purities = dt.GetPurities();
    EXPECT_FLOAT_EQ( purities[0], 0.5 );
    EXPECT_FLOAT_EQ( purities[1], 0.875 );
    EXPECT_FLOAT_EQ( purities[2], 0.25 );
    EXPECT_FLOAT_EQ( purities[3], 1.0 );
    EXPECT_FLOAT_EQ( purities[4], 0.5 );
    EXPECT_FLOAT_EQ( purities[5], 0.42857142857142855 );
    EXPECT_FLOAT_EQ( purities[6], 0.0 );

}

TEST_F(TreeBuilderTest, BoostWeightsOfNodesAreCorrectAfterTraining) {
    
    TreeBuilder dt(2, *eventSample);
    const auto &boostWeights = dt.GetBoostWeights();
    EXPECT_FLOAT_EQ( boostWeights[0], 0.0 );
    EXPECT_FLOAT_EQ( boostWeights[1], -1.0 );
    EXPECT_FLOAT_EQ( boostWeights[2], 0.42857142857142855 );
    EXPECT_FLOAT_EQ( boostWeights[3], -0.75 );
    EXPECT_FLOAT_EQ( boostWeights[4], 0 );
    EXPECT_FLOAT_EQ( boostWeights[5], 0.090909090909090912 );
    EXPECT_FLOAT_EQ( boostWeights[6], 1.6666666666666667 );

}


class TreeTest : public ::testing::Test {
    protected:
        virtual void SetUp() {

            Cut<unsigned int> cut1, cut2, cut3;
            cut1.feature = 0;
            cut1.index = 5;
            cut1.valid = true;
            cut2.feature = 1;
            cut2.index = 9;
            cut2.valid = true;
            cut3.valid = false;
            
            std::vector<Cut<unsigned int>> cuts = {cut1, cut2, cut3};
            std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
            std::vector<Weight> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
            std::vector<Weight> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            tree = new Tree<unsigned int>(cuts, nEntries, purities, boostWeights);            
        }

        virtual void TearDown() {
            delete tree;
        }

        Tree<unsigned int> *tree;

};

TEST_F(TreeTest, ValueToNode) {

    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,3,31}) ), 3u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,9,4}) ), 4u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({4,9,31}) ), 4u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({4,8,4}) ), 3u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({5,8,31}) ), 2u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({5,9,4}) ), 2u );

}

TEST_F(TreeTest, NaNToNode) {

    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({0,3,31}) ), 0u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,3,0}) ), 3u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,0,4}) ), 1u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,9,4}) ), 4u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({5,0,31}) ), 2u );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({5,9,0}) ), 2u );

}

TEST_F(TreeTest, NEntries) {
    for(unsigned int i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(tree->GetNEntries(i), 10.0 + i);
    }
}

TEST_F(TreeTest, Purities) {
    for(unsigned int i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(tree->GetPurity(i), 0.1*(i+1) );
    }
}

TEST_F(TreeTest, BoostWeights) {
    for(unsigned int i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(tree->GetBoostWeight(i), 1.0*(i+1) );
    }
}

class ForestBuilderTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(20, 2, {1, 1});
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, true);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
            eventSample->AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;
};

TEST_F(ForestBuilderTest, F0AndShrinkageIsCorrect) {

    // Train without randomness and only with one layer per tree
    ForestBuilder forest(*eventSample, 0, 0.1, 1.0, 1); 
    EXPECT_FLOAT_EQ(forest.GetF0(), 0);
    EXPECT_FLOAT_EQ(forest.GetShrinkage(), 0.1);

} 

TEST_F(ForestBuilderTest, ForestIsCorrect) {

    // Train without randomness and only with one layer per tree
    ForestBuilder forest(*eventSample, 5, 0.1, 1.0, 1); 
    auto trees = forest.GetForest();
    EXPECT_EQ(trees[0].GetCut(0).feature, 0u);
    EXPECT_EQ(trees[1].GetCut(0).feature, 0u);
    EXPECT_EQ(trees[2].GetCut(0).feature, 1u);
    EXPECT_EQ(trees[3].GetCut(0).feature, 0u);
    EXPECT_EQ(trees[4].GetCut(0).feature, 1u);

}

class ForestTest : public ::testing::Test {
    protected:
        virtual void SetUp() {

            Cut<unsigned int> cut1, cut2, cut3;
            cut1.feature = 0;
            cut1.index = 5;
            cut1.valid = true;
            cut1.gain = 2.0;
            cut2.feature = 1;
            cut2.index = 9;
            cut2.valid = true;
            cut2.gain = 1.0;
            cut3.valid = false;
            
            std::vector<Cut<unsigned int>> cuts = {cut1, cut2, cut3};
            std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
            std::vector<Weight> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
            std::vector<Weight> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            tree = new Tree<unsigned int>(cuts, nEntries, purities, boostWeights);            

            forest = new Forest<unsigned int>(0.1, 1.0, true);
        }

        virtual void TearDown() {
            delete tree;
            delete forest;
        }

        Tree<unsigned int> *tree;
        Forest<unsigned int> *forest;

};

TEST_F(ForestTest, GetF) {

    std::vector<unsigned int> values = {1,1};
    EXPECT_FLOAT_EQ(forest->GetF(values), 1.0);
    forest->AddTree(*tree);
    EXPECT_FLOAT_EQ(forest->GetF(values), 1.4);
    forest->AddTree(*tree);
    EXPECT_FLOAT_EQ(forest->GetF(values), 1.8);

}


class VariableRankingTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            forest = new Forest<unsigned int>(0.1, 1.0, true);
        }

        virtual void TearDown() {
            delete forest;
        }

        Forest<unsigned int> *forest;
};

TEST_F(VariableRankingTest, OneVariable) {
    
    Cut<unsigned int> cut1;
    cut1.feature = 1;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = 2.0;

    std::vector<Cut<unsigned int>> cuts = {cut1};
    std::vector<Weight> nEntries = { 10.0, 4.0, 6.0};
    std::vector<Weight> purities = { 0.1, 0.2, 0.3 };
    std::vector<Weight> boostWeights = {1.0, 2.0, 3.0};
    Tree<unsigned int> tree(cuts, nEntries, purities, boostWeights);            
    forest->AddTree(tree);
    auto map = forest->GetVariableRanking();
    EXPECT_EQ(map.size(), 1u);
    EXPECT_FLOAT_EQ(map[1], 1.0);

} 

TEST_F(VariableRankingTest, Standard) {
    
    Cut<unsigned int> cut1, cut2, cut3;
    cut1.feature = 0;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = 2.0;
    cut2.feature = 1;
    cut2.index = 9;
    cut2.valid = true;
    cut2.gain = 1.0;
    cut3.valid = false;

    std::vector<Cut<unsigned int>> cuts = {cut1, cut2, cut3};
    std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    std::vector<Weight> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
    std::vector<Weight> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tree<unsigned int> tree(cuts, nEntries, purities, boostWeights);            
    forest->AddTree(tree);
    auto map = forest->GetVariableRanking();
    EXPECT_FLOAT_EQ(map[0], 2.0/3.0);
    EXPECT_FLOAT_EQ(map[1], 1.0/3.0);

} 

TEST_F(VariableRankingTest, Individual) {
    
    Cut<unsigned int> cut1, cut2, cut3;
    cut1.feature = 0;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = 2.0;
    cut2.feature = 1;
    cut2.index = 9;
    cut2.valid = true;
    cut2.gain = 1.0;
    cut3.valid = false;

    std::vector<Cut<unsigned int>> cuts = {cut1, cut2, cut3};
    std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    std::vector<Weight> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
    std::vector<Weight> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tree<unsigned int> tree(cuts, nEntries, purities, boostWeights);            
    forest->AddTree(tree);
    std::vector<unsigned int> event = {4, 8};
    auto map = forest->GetIndividualVariableRanking(event);
    EXPECT_FLOAT_EQ(map[0], 2.0/3.0);
    EXPECT_FLOAT_EQ(map[1], 1.0/3.0);
    
    std::vector<unsigned int> event2 = {6, 8};
    auto map2 = forest->GetIndividualVariableRanking(event2);
    EXPECT_FLOAT_EQ(map2[0], 1.0);
    EXPECT_FLOAT_EQ(map2[1], 0.0);

} 


class CornerCasesTest : public ::testing::Test { };

TEST_F(CornerCasesTest, OnlySignalGivesReasonableResult) {

    EventSample eventSample(5, 2, {1, 1});
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, true);
    // Train without randomness and only with one layer per tree
    ForestBuilder forest(eventSample, 10, 0.1, 1.0, 1); 
    EXPECT_FLOAT_EQ(forest.GetF0(), std::numeric_limits<double>::infinity());
    
    FastBDT::Forest<unsigned int> testforest( forest.GetShrinkage(), forest.GetF0(), true);
    for( auto t : forest.GetForest() )
        testforest.AddTree(t);
    
    std::vector<unsigned int> values = {0, 1};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 1.0);
    
    values = {2, 1};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 1.0);
    
    // Even for NaN values the signal probability should be one
    values = {0, 0};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 1.0);

}

TEST_F(CornerCasesTest, OnlyBackgroundGivesReasonableResult) {

    EventSample eventSample(5, 2, {1, 1});
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, false);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, false);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, false);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, false);
    eventSample.AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, false);
    // Train without randomness and only with one layer per tree
    ForestBuilder forest(eventSample, 10, 0.1, 1.0, 1); 
    EXPECT_FLOAT_EQ(forest.GetF0(), -std::numeric_limits<double>::infinity());
    
    FastBDT::Forest<unsigned int> testforest( forest.GetShrinkage(), forest.GetF0(), true);
    for( auto t : forest.GetForest() )
        testforest.AddTree(t);
    
    std::vector<unsigned int> values = {0, 1};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 0.0);
    
    values = {2, 1};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 0.0);
    
    // Even for NaN values the signal probability should be zero
    values = {0, 0};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 0.0);

}

TEST_F(CornerCasesTest, PerfectSeparationWithDifferentWeights) {
    
    EventSample eventSample1(6, 2, {1, 1});
    eventSample1.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample1.AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, true);
    eventSample1.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample1.AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
    eventSample1.AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, false);
    eventSample1.AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);

    EventSample eventSample2(6, 2, {1, 1});
    eventSample2.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample2.AddEvent( std::vector<unsigned int>({ 2, 1 }), 2.0, true);
    eventSample2.AddEvent( std::vector<unsigned int>({ 1, 1 }), 3.0, true);
    eventSample2.AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
    eventSample2.AddEvent( std::vector<unsigned int>({ 1, 2 }), 2.0, false);
    eventSample2.AddEvent( std::vector<unsigned int>({ 2, 2 }), 3.0, false);

    EventSample eventSample3(7, 2, {1, 1});
    eventSample3.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample3.AddEvent( std::vector<unsigned int>({ 2, 1 }), 1.0, true);
    eventSample3.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample3.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample3.AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
    eventSample3.AddEvent( std::vector<unsigned int>({ 1, 2 }), 1.0, false);
    eventSample3.AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);

    EventSample eventSample4(7, 2, {1, 1});
    eventSample4.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample4.AddEvent( std::vector<unsigned int>({ 2, 1 }), 2.0, true);
    eventSample4.AddEvent( std::vector<unsigned int>({ 1, 1 }), 3.0, true);
    eventSample4.AddEvent( std::vector<unsigned int>({ 1, 1 }), 4.0, true);
    eventSample4.AddEvent( std::vector<unsigned int>({ 2, 2 }), 1.0, false);
    eventSample4.AddEvent( std::vector<unsigned int>({ 1, 2 }), 2.0, false);
    eventSample4.AddEvent( std::vector<unsigned int>({ 2, 2 }), 3.0, false);

    EventSample eventSample5(7, 2, {1, 1});
    eventSample5.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample5.AddEvent( std::vector<unsigned int>({ 2, 1 }), 2.0, true);
    eventSample5.AddEvent( std::vector<unsigned int>({ 1, 1 }), 3.0, true);
    eventSample5.AddEvent( std::vector<unsigned int>({ 1, 1 }), 1.0, true);
    eventSample5.AddEvent( std::vector<unsigned int>({ 2, 2 }), 2.0, false);
    eventSample5.AddEvent( std::vector<unsigned int>({ 1, 2 }), 3.0, false);
    eventSample5.AddEvent( std::vector<unsigned int>({ 2, 2 }), 2.0, false);

    std::vector<EventSample*> eventSamples = {&eventSample1, &eventSample2, &eventSample3, &eventSample4, &eventSample5};

    for(auto &sample : eventSamples) {

        // Calculate prior probability before building the forest
        Weight sig = 0;
        Weight tot = 0;
        const unsigned int nSignals = sample->GetNSignals();
        const unsigned int nEvents = sample->GetNEvents();
        for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
            if(iEvent < nSignals)
                sig += sample->GetWeights().GetOriginal(iEvent);
            tot += sample->GetWeights().GetOriginal(iEvent);
        }

        // Train without randomness and only with one layer per tree and shrinkage 1
        // We won't get a perfect separation due to the regularisation of the boostWeights,
        // however with 10 trees we already get pretty close to perfect separation.
        ForestBuilder forest(*sample, 10, 1.0, 1.0, 1); 
        
        FastBDT::Forest<unsigned int> testforest( forest.GetShrinkage(), forest.GetF0(), true);
        for( auto t : forest.GetForest() ) {
            //t.Print();
            testforest.AddTree(t);
        }
        
        std::vector<unsigned int> values = {0, 1};
        EXPECT_GE(testforest.Analyse(values), 0.999);
        
        values = {2, 1};
        EXPECT_GE(testforest.Analyse(values), 0.999);
        
        values = {0, 2};
        EXPECT_LE(testforest.Analyse(values), 0.001);
        
        values = {1, 2};
        EXPECT_LE(testforest.Analyse(values), 0.001);
        
        // Even for NaN values the signal probability should be zero
        values = {0, 0};
        EXPECT_FLOAT_EQ(testforest.Analyse(values), sig/tot);
    }
}

TEST_F(CornerCasesTest, PerfectSeparationGivesReasonableResults) {

    Cut<unsigned int> cut1;
    cut1.feature = 0;
    cut1.index = 2;
    cut1.valid = true;
    
    std::vector<Cut<unsigned int>> cuts = {cut1};
    std::vector<Weight> nEntries = { 10.0, 11.0, 12.0 };
    std::vector<Weight> purities = { 0.5, 0.0, 1.0};
    std::vector<Weight> boostWeights = { 0.0, -std::numeric_limits<Weight>::infinity(), std::numeric_limits<Weight>::infinity()};
    Tree<unsigned int> testtree(cuts, nEntries, purities, boostWeights);
    Forest<unsigned int> testforest(0.1, 0.0, true);
    testforest.AddTree(testtree);

    std::vector<unsigned int> values = {1, 1};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 0.0);
    
    values = {3, 1};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 1.0);
    
    values = {0, 0};
    EXPECT_FLOAT_EQ(testforest.Analyse(values), 0.5);

} 

class RegressionTest : public ::testing::Test {

};

TEST_F(TreeTest, LastCutOnTheRightWasNeverUsed) {
            
    Cut<unsigned int> cut1, cut2, cut3;
    cut1.feature = 0;
    cut1.index = 2;
    cut1.valid = true;
    cut2.feature = 1;
    cut2.index = 2;
    cut2.valid = true;
    cut3.feature = 1;
    cut3.index = 2;
    cut3.valid = true;
    
    std::vector<Cut<unsigned int>> cuts = {cut1, cut2, cut3};
    std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    std::vector<Weight> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
    std::vector<Weight> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tree<unsigned int> tree(cuts, nEntries, purities, boostWeights);            

    // Check if we can reach all nodes
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({0,0}) ), 0u );
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({1,0}) ), 1u );
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({2,0}) ), 2u );
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({1,1}) ), 3u );
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({1,2}) ), 4u );
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({2,1}) ), 5u );
    EXPECT_EQ(tree.ValueToNode( std::vector<unsigned int>({2,2}) ), 6u );

}

class RewriteTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            
            std::vector<float> binning = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }; 
            featureBinning = new FeatureBinning<float>(2, binning);

            Cut<unsigned int> cut1, cut2, cut3;
            cut1.feature = 0;
            cut1.index = 2;
            cut1.valid = true;
            cut1.gain = 1.0;
            cut2.feature = 0;
            cut2.index = 1;
            cut2.valid = true;
            cut2.gain = 1.0;
            cut3.feature = 0;
            cut3.index = 4;
            cut3.valid = true;
            cut3.gain = 1.0;
            
            std::vector<Cut<unsigned int>> cuts = {cut1, cut2, cut3};
            std::vector<Weight> nEntries = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
            std::vector<Weight> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
            std::vector<Weight> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            Tree<unsigned int> tree(cuts, nEntries, purities, boostWeights);
            forest = new Forest<unsigned int>(1.0, 0.0, true);
            forest->AddTree(tree);
        }

        virtual void TearDown() {
            delete forest;
            delete featureBinning;
        }
        FeatureBinning<float> *featureBinning;
        Forest<unsigned int> *forest;

};

TEST_F(RewriteTest, CheckSameResultForOriginalAndRewrittenFloatForest) {

    auto rewritten_forest = removeFeatureBinningTransformationFromForest<float>(*forest, {*featureBinning});
    std::vector<float> values = {-1.0f, -0.5f, 0.0f, 0.1f, 0.25f, 0.3f, 0.5f, 0.6f, 0.75f, 0.9f, 1.0f, 1.2f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), NAN, std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
    for(auto &x : values) {
      EXPECT_FLOAT_EQ(forest->GetF(std::vector<unsigned int>({featureBinning->ValueToBin(x)})), rewritten_forest.GetF(std::vector<float>({x})));
    }

}
