/**
 * Thomas Keck 2014
 */

#include "FBDT.h"
#include "IO.h"

#include <gtest/gtest.h>

#include <sstream>

using namespace FastBDT;

class FeatureBinningTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            std::vector<float> data = {10.0f,8.0f,2.0f,7.0f,5.0f,6.0f,9.0f,4.0f,3.0f,11.0f,12.0f,1.0f};
            calculatedBinning = new FeatureBinning<float>(2, data.begin(), data.end());

            binning = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
            predefinedBinning = new FeatureBinning<float>(2, binning.begin(), binning.end());
            
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

    EXPECT_DOUBLE_EQ( calculatedBinning->GetMin(), 1.0f);
    EXPECT_DOUBLE_EQ( calculatedBinning->GetMax(), 12.0f);
    EXPECT_DOUBLE_EQ( predefinedBinning->GetMin(), 1.0f);
    EXPECT_DOUBLE_EQ( predefinedBinning->GetMax(), 12.0f);

}

TEST_F(FeatureBinningTest, NumberOfLevelsAndBinsIsCorrectlyIdentified) {

    EXPECT_EQ( calculatedBinning->GetNLevels(), 2 );
    EXPECT_EQ( predefinedBinning->GetNLevels(), 2 );
    // 5 bins, 2^2 ordinary bins + 1 NaN bin
    EXPECT_EQ( calculatedBinning->GetNBins(), 5 );
    EXPECT_EQ( predefinedBinning->GetNBins(), 5 );

}

TEST_F(FeatureBinningTest, ValueToBinMapsNormalValuesCorrectly) {

    EXPECT_EQ( calculatedBinning->ValueToBin(1.0f), 1);
    EXPECT_EQ( calculatedBinning->ValueToBin(2.0f), 1);
    EXPECT_EQ( calculatedBinning->ValueToBin(3.0f), 1);
    EXPECT_EQ( calculatedBinning->ValueToBin(4.0f), 2);
    EXPECT_EQ( calculatedBinning->ValueToBin(5.0f), 2);
    EXPECT_EQ( calculatedBinning->ValueToBin(6.0f), 2);
    EXPECT_EQ( calculatedBinning->ValueToBin(7.0f), 3);
    EXPECT_EQ( calculatedBinning->ValueToBin(8.0f), 3);
    EXPECT_EQ( calculatedBinning->ValueToBin(9.0f), 3);
    EXPECT_EQ( calculatedBinning->ValueToBin(10.0f), 4);
    EXPECT_EQ( calculatedBinning->ValueToBin(11.0f), 4);
    EXPECT_EQ( calculatedBinning->ValueToBin(12.0f), 4);
    
    EXPECT_EQ( predefinedBinning->ValueToBin(1.0f), 1);
    EXPECT_EQ( predefinedBinning->ValueToBin(2.0f), 1);
    EXPECT_EQ( predefinedBinning->ValueToBin(3.0f), 1);
    EXPECT_EQ( predefinedBinning->ValueToBin(4.0f), 2);
    EXPECT_EQ( predefinedBinning->ValueToBin(5.0f), 2);
    EXPECT_EQ( predefinedBinning->ValueToBin(6.0f), 2);
    EXPECT_EQ( predefinedBinning->ValueToBin(7.0f), 3);
    EXPECT_EQ( predefinedBinning->ValueToBin(8.0f), 3);
    EXPECT_EQ( predefinedBinning->ValueToBin(9.0f), 3);
    EXPECT_EQ( predefinedBinning->ValueToBin(10.0f), 4);
    EXPECT_EQ( predefinedBinning->ValueToBin(11.0f), 4);
    EXPECT_EQ( predefinedBinning->ValueToBin(12.0f), 4);

}

TEST_F(FeatureBinningTest, NaNGivesZeroBin) {

    EXPECT_EQ( predefinedBinning->ValueToBin(NAN), 0);
    EXPECT_EQ( predefinedBinning->ValueToBin(NAN), 0);

}

TEST_F(FeatureBinningTest, OverflowAndUnderflowGivesLastAndFirstBin) {

    EXPECT_EQ( calculatedBinning->ValueToBin(100.0f), 4);
    EXPECT_EQ( calculatedBinning->ValueToBin(-100.0f), 1);
    EXPECT_EQ( predefinedBinning->ValueToBin(100.0f), 4);
    EXPECT_EQ( predefinedBinning->ValueToBin(-100.0f), 1);

}

TEST_F(FeatureBinningTest, GetBinningIsCorrect) {

    EXPECT_EQ( calculatedBinning->GetBinning(), binning);
    EXPECT_EQ( predefinedBinning->GetBinning(), binning);

}

TEST_F(FeatureBinningTest, ThrowIfTooManyLevelsForGivenData) {

    std::vector<float> data = { 1.0f, 4.0f, 7.0f, 10.0f, 12.0f }; 
    EXPECT_THROW( FeatureBinning<float>(3, data.begin(), data.end()), std::runtime_error );

}

class EventWeightsTest : public ::testing::Test {

    protected:
        virtual void SetUp() {
            eventWeights = new EventWeights(10);
            for(unsigned int i = 0; i < 10; ++i) {
                eventWeights->Set(i, static_cast<float>(i+1));
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
    EXPECT_DOUBLE_EQ(sums[0], 15.0 * 2);
    EXPECT_DOUBLE_EQ(sums[1], 40.0 * 2);
    EXPECT_DOUBLE_EQ(sums[2], 385.0 * 2);

}

TEST_F(EventWeightsTest, GetterIsCorrect) {

    for(unsigned int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ( eventWeights->Get(i), static_cast<float>(i+1) * 2); 
    }
    
}

TEST_F(EventWeightsTest, WeightSumsAndGetterAreCorrectlyUpdated) {

    for(unsigned int i = 0; i < 10; ++i) {
        eventWeights->Set(i, static_cast<float>(i+3));
    }

    auto sums = eventWeights->GetSums(5);
    EXPECT_DOUBLE_EQ(sums[0], 25.0 * 2);
    EXPECT_DOUBLE_EQ(sums[1], 50.0 * 2);
    EXPECT_DOUBLE_EQ(sums[2], 645.0 * 2);
    
    for(unsigned int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ( eventWeights->Get(i), static_cast<float>(i+3) * 2); 
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
        EXPECT_EQ( eventFlags->Get(i), i-5);

}

class EventValuesTest : public ::testing::Test {

    protected:
        virtual void SetUp() {
            eventValues = new EventValues(8, 3, 3);
        }

        virtual void TearDown() {
            delete eventValues;
        }

        EventValues *eventValues;
};

TEST_F(EventValuesTest, SetterAndGetterWorkCorrectly) {
    
    std::vector<unsigned int> features = { 1, 2, 3 };
    for(unsigned int i = 0; i < 8; ++i) {
        std::vector<unsigned int> features = { i, 7-i, static_cast<unsigned int>(4 + (1-2*((int)i%2))*((int)i+1)/2) };
        eventValues->Set(i, features);
    }
    EXPECT_THROW( eventValues->Set(1, {1,2,3,4}), std::runtime_error );
    EXPECT_THROW( eventValues->Set(1, {1,20,3}), std::runtime_error );
    
    for(unsigned int i = 0; i < 8; ++i) {
        std::vector<unsigned int> features = { i, 7-i, static_cast<unsigned int>(4 + (1-2*((int)(i)%2))*((int)(i)+1)/2) };
        const auto *array = &eventValues->Get(i);
        for(unsigned int j = 0; j < 3; ++j) {
            EXPECT_EQ( eventValues->Get(i,j), features[j]);
            EXPECT_EQ( array[j], features[j]);
        }
    }
}

TEST_F(EventValuesTest, GetSizesWorkCorrectly) {

    EXPECT_EQ( eventValues->GetNFeatures(), 3);
    EXPECT_EQ( eventValues->GetNBins(), 9);

}

class EventSampleTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(10, 3, 8);
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;

};

TEST_F(EventSampleTest, AddingEventsWorksCorrectly) {

    eventSample->AddEvent( std::vector<unsigned int>({1,2,3}), 2.0, true );
    EXPECT_EQ( eventSample->GetNSignals(), 1);
    EXPECT_EQ( eventSample->GetNBckgrds(), 0);
   
    const auto &eventWeights = eventSample->GetWeights();
    auto sums = eventWeights.GetSums(5);
    EXPECT_DOUBLE_EQ(sums[0], 2.0);
    EXPECT_DOUBLE_EQ(sums[1], 0.0);

 
    // Add some more Signal and Background events   
    for(unsigned int i = 1; i < 10; ++i) { 
        eventSample->AddEvent( std::vector<unsigned int>({2*i,3*i,5*i}), 2.0, i % 2 == 0 );
    }
    EXPECT_EQ( eventSample->GetNSignals(), 5);
    EXPECT_EQ( eventSample->GetNBckgrds(), 5);
    
    sums = eventWeights.GetSums(5);
    EXPECT_DOUBLE_EQ(sums[0], 10.0);
    EXPECT_DOUBLE_EQ(sums[1], 10.0);
    
    // Test throw if event with 0 weight is added
    EXPECT_THROW( eventSample->AddEvent( std::vector<unsigned int>({1,2,3}), 0.0, true ), std::runtime_error);

    // Test some of the values, if they're correct
    // Remember that the events are NOT in the same order as they were added,
    // instead the signal events are added starting from 0, and the background events
    // are added reversed starting from the last event.
    EXPECT_EQ( eventSample->GetValues().Get(1,2), 10); 
    EXPECT_EQ( eventSample->GetValues().Get(3,1), 18); 
    EXPECT_EQ( eventSample->GetValues().Get(9,0), 2); 

    // Test if signal and background labels are correctly assigned
    for(unsigned int i = 0; i < 5; ++i) {
        EXPECT_TRUE( eventSample->IsSignal(i));
        EXPECT_FALSE( eventSample->IsSignal(i+5));
    }

    // Test throw if number of promised events is exceeded
    EXPECT_THROW( eventSample->AddEvent( std::vector<unsigned int>({1,2,3}), 2.0, true ), std::runtime_error);
    
}

class CumulativeDistributionsTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            const unsigned int numberOfEvents = 100;
            eventSample = new EventSample(numberOfEvents, 2, 2);
            for(unsigned int i = 0; i < numberOfEvents; ++i) {
                bool isSignal = i < (numberOfEvents/2);
                eventSample->AddEvent( std::vector<unsigned int>({i % 4 + 1, (numberOfEvents-i) % 4 + 1}), static_cast<float>(i+1), isSignal);
            }
        }

        virtual void TearDown() {
            delete eventSample;
        }

        EventSample *eventSample;
};

TEST_F(CumulativeDistributionsTest, CheckIfLayer0IsCorrect) {

    CumulativeDistributions CDFsForLayer0(0, *eventSample);

    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 0, 1), 325.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 0, 2), 663.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 0, 3), 963.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 0, 4), 1275.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 1, 1), 325.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 1, 2), 637.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 1, 3), 937.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetSignal(0, 1, 4), 1275.0); 

    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 0, 1), 900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 0, 2), 1812.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 0, 3), 2787.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 0, 4), 3775.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 1, 1), 900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 1, 2), 1888.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 1, 3), 2863.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer0.GetBckgrd(0, 1, 4), 3775.0); 

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

    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 0, 1), 325.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 0, 2), 325.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 0, 3), 625.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 0, 4), 625.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 1, 1), 325.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 1, 2), 325.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 1, 3), 625.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(0, 1, 4), 625.0); 

    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 0, 1), 900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 0, 2), 900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 0, 3), 1875.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 0, 4), 1875.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 1, 1), 900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 1, 2), 900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 1, 3), 1875.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(0, 1, 4), 1875.0); 

    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 0, 1), 0.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 0, 2), 338.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 0, 3), 338.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 0, 4), 650.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 1, 1), 0.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 1, 2), 312.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 1, 3), 312.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetSignal(1, 1, 4), 650.0); 

    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 0, 1), 0.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 0, 2), 912.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 0, 3), 912.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 0, 4), 1900.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 1, 1), 0.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 1, 2), 988.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 1, 3), 988.0); 
    EXPECT_DOUBLE_EQ( CDFsForLayer1.GetBckgrd(1, 1, 4), 1900.0); 

}

class LossFunctionTest : public ::testing::Test { };

TEST_F(LossFunctionTest, GiniIndexIsCorrect) {

    EXPECT_DOUBLE_EQ( LossFunction(4,4), 2.0);
    EXPECT_DOUBLE_EQ( LossFunction(1,4), 0.8);
    EXPECT_DOUBLE_EQ( LossFunction(4,1), 0.8);
    EXPECT_DOUBLE_EQ( LossFunction(2,0), 0.0);
    EXPECT_DOUBLE_EQ( LossFunction(0,2), 0.0);

}

class NodeTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(8, 2, 1);
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

    EXPECT_EQ( Node(0,0).GetPosition(), 0 );
    EXPECT_EQ( Node(1,0).GetPosition(), 1 );
    EXPECT_EQ( Node(1,1).GetPosition(), 2 );
    EXPECT_EQ( Node(2,0).GetPosition(), 3 );
    EXPECT_EQ( Node(2,1).GetPosition(), 4 );
    EXPECT_EQ( Node(2,2).GetPosition(), 5 );
    EXPECT_EQ( Node(2,3).GetPosition(), 6 );
    EXPECT_EQ( Node(3,0).GetPosition(), 7 );
    EXPECT_EQ( Node(3,7).GetPosition(), 14 );

}

TEST_F(NodeTest, BoostWeightCalculation) {

    Node node(0,0);
    node.SetWeights({2.0, 2.0, 4.0});
    EXPECT_DOUBLE_EQ(node.GetBoostWeight(), 0.0); 
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(1.0, 1.0);
    node.AddSignalWeight(3.0, 1.0);
    node.AddBckgrdWeight(2.0, 1.0);
    EXPECT_DOUBLE_EQ(node.GetBoostWeight(), -1.0);

}

TEST_F(NodeTest, PurityCalculation) {

    Node node(0,0);
    node.SetWeights({2.0, 2.0, 4.0});
    EXPECT_DOUBLE_EQ(node.GetPurity(), 0.5); 
    node.SetWeights({0.0, 0.0, 0.0});
    node.AddSignalWeight(2.0, 1.0);
    node.AddSignalWeight(4.0, 1.0);
    node.AddBckgrdWeight(4.0, 1.0);
    EXPECT_DOUBLE_EQ(node.GetPurity(), 0.6);

}

TEST_F(NodeTest, BestCut0Layer) {

    CumulativeDistributions CDFs(0, *eventSample);
    Node node(0,0);
    node.SetWeights({10.0, 10.0, 68.0});

    Cut bestCut = node.CalculateBestCut(CDFs);
    EXPECT_EQ( bestCut.feature, 0 );
    EXPECT_EQ( bestCut.index, 2 );
    EXPECT_DOUBLE_EQ( bestCut.gain, 1.875 );
    EXPECT_TRUE( bestCut.valid );

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
    Cut right_bestCut = right_node.CalculateBestCut(CDFs);
    EXPECT_EQ( right_bestCut.feature, 1 );
    EXPECT_EQ( right_bestCut.index, 2 );
    EXPECT_DOUBLE_EQ( right_bestCut.gain, 0.375);
    EXPECT_TRUE( right_bestCut.valid );
    
    Node left_node(1,1);
    left_node.SetWeights({3.0, 9.0, 38.0});
    Cut left_bestCut = left_node.CalculateBestCut(CDFs);
    EXPECT_EQ( left_bestCut.feature, 1 );
    EXPECT_EQ( left_bestCut.index, 2 );
    EXPECT_DOUBLE_EQ( left_bestCut.gain, 0.53571428571428581);
    EXPECT_TRUE( left_bestCut.valid );

}

class TreeBuilderTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            eventSample = new EventSample(8, 2, 1);
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
    EXPECT_EQ( cuts[0].feature, 0 );
    EXPECT_EQ( cuts[0].index, 2 );
    EXPECT_DOUBLE_EQ( cuts[0].gain, 1.875 );
    EXPECT_TRUE( cuts[0].valid );
    
    EXPECT_EQ( cuts[1].feature, 1 );
    EXPECT_EQ( cuts[1].index, 2 );
    EXPECT_DOUBLE_EQ( cuts[1].gain, 0.375 );
    EXPECT_TRUE( cuts[1].valid );
    
    EXPECT_EQ( cuts[2].feature, 1 );
    EXPECT_EQ( cuts[2].index, 2 );
    EXPECT_DOUBLE_EQ( cuts[2].gain, 0.53571428571428581 );
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


TEST_F(TreeBuilderTest, PuritiesOfNodesAreCorrectAfterTraining) {
    
    TreeBuilder dt(2, *eventSample);
    const auto &purities = dt.GetPurities();
    EXPECT_DOUBLE_EQ( purities[0], 0.5 );
    EXPECT_DOUBLE_EQ( purities[1], 0.875 );
    EXPECT_DOUBLE_EQ( purities[2], 0.25 );
    EXPECT_DOUBLE_EQ( purities[3], 1.0 );
    EXPECT_DOUBLE_EQ( purities[4], 0.5 );
    EXPECT_DOUBLE_EQ( purities[5], 0.4285714328289032 );
    EXPECT_DOUBLE_EQ( purities[6], 0.0 );

}

TEST_F(TreeBuilderTest, BoostWeightsOfNodesAreCorrectAfterTraining) {
    
    TreeBuilder dt(2, *eventSample);
    const auto &boostWeights = dt.GetBoostWeights();
    EXPECT_DOUBLE_EQ( boostWeights[0], 0.0 );
    EXPECT_DOUBLE_EQ( boostWeights[1], -1.0 );
    EXPECT_DOUBLE_EQ( boostWeights[2], 0.4285714328289032 );
    EXPECT_DOUBLE_EQ( boostWeights[3], -0.75 );
    EXPECT_DOUBLE_EQ( boostWeights[4], 0 );
    EXPECT_DOUBLE_EQ( boostWeights[5], 0.090909093618392944 );
    EXPECT_DOUBLE_EQ( boostWeights[6], 1.6666666269302368 );

}


class TreeTest : public ::testing::Test {
    protected:
        virtual void SetUp() {

            Cut cut1, cut2, cut3;
            cut1.feature = 0;
            cut1.index = 5;
            cut1.valid = true;
            cut2.feature = 1;
            cut2.index = 9;
            cut2.valid = true;
            cut3.valid = false;
            
            std::vector<Cut> cuts = {cut1, cut2, cut3};
            std::vector<float> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
            std::vector<float> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            tree = new Tree(cuts, purities, boostWeights);            
        }

        virtual void TearDown() {
            delete tree;
        }

        Tree *tree;

};

TEST_F(TreeTest, ValueToNode) {

    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,3,31}) ), 3 );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({2,9,4}) ), 4 );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({4,9,31}) ), 4 );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({4,8,4}) ), 3 );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({5,8,31}) ), 2 );
    EXPECT_EQ(tree->ValueToNode( std::vector<unsigned int>({5,9,4}) ), 2 );

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
            eventSample = new EventSample(20, 2, 1);
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
    EXPECT_FLOAT_EQ(trees[0].GetCut(0).feature, 0);
    EXPECT_FLOAT_EQ(trees[1].GetCut(0).feature, 0);
    EXPECT_FLOAT_EQ(trees[2].GetCut(0).feature, 1);
    EXPECT_FLOAT_EQ(trees[3].GetCut(0).feature, 0);
    EXPECT_FLOAT_EQ(trees[4].GetCut(0).feature, 1);

}

class ForestTest : public ::testing::Test {
    protected:
        virtual void SetUp() {

            Cut cut1, cut2, cut3;
            cut1.feature = 0;
            cut1.index = 5;
            cut1.valid = true;
            cut1.gain = 2.0;
            cut2.feature = 1;
            cut2.index = 9;
            cut2.valid = true;
            cut2.gain = 1.0;
            cut3.valid = false;
            
            std::vector<Cut> cuts = {cut1, cut2, cut3};
            std::vector<float> purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
            std::vector<float> boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            tree = new Tree(cuts, purities, boostWeights);            

            forest = new Forest(0.1, 1.0);
        }

        virtual void TearDown() {
            delete tree;
            delete forest;
        }

        Tree *tree;
        Forest *forest;

};

TEST_F(ForestTest, GetF) {

    std::vector<unsigned int> values = {1,1};
    EXPECT_FLOAT_EQ(forest->GetF(values), 1.0);
    forest->AddTree(*tree);
    EXPECT_FLOAT_EQ(forest->GetF(values), 1.4);
    forest->AddTree(*tree);
    EXPECT_FLOAT_EQ(forest->GetF(values), 1.8);

}

TEST_F(ForestTest, VariableRankingIsCorrect) {

    // Train without randomness and only with one layer per tree
    forest->AddTree(*tree);
    auto map = forest->GetVariableRanking();
    EXPECT_FLOAT_EQ(map[0], 2.0);
    EXPECT_FLOAT_EQ(map[1], 2.0);

} 

class IOTest : public ::testing::Test {
    protected:
        virtual void SetUp() {}
        virtual void TearDown() {}

};

TEST_F(IOTest, IOVector) {

    std::vector<double> before = {0.0, 1.0, 2.5, 3.2, -1.4, 0.0};

    std::stringstream stream;
    stream << before;

    std::vector<double> after;
    stream >> after;

    EXPECT_EQ(before.size(), after.size());
    for(unsigned int i = 0; i < before.size() and i < after.size(); ++i)
        EXPECT_DOUBLE_EQ(before[i], after[i]);

}

TEST_F(IOTest, IOFeatureBinning) {

    std::vector<double> binning = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
    FeatureBinning<double> before(2, binning.begin(), binning.end());
    const auto &before_binning = before.GetBinning();

    std::stringstream stream;
    stream << before;

    auto after = readFeatureBinningFromStream<double>(stream);
    const auto &after_binning = after.GetBinning();

    EXPECT_EQ(before.GetNLevels(), after.GetNLevels());
    EXPECT_EQ(before_binning.size(), after_binning.size());
    for(unsigned int i = 0; i < before_binning.size() and i < after_binning.size(); ++i)
        EXPECT_DOUBLE_EQ(before_binning[i], after_binning[i]);

}

TEST_F(IOTest, IOFeatureBinningVector) {

    std::vector<double> binning1 = { 1.0f, 7.0f, 4.0f, 10.0f, 12.0f }; 
    std::vector<double> binning2 = { 6.0f, 7.0f, 2.0f, 12.0f, 12.0f }; 
    std::vector<FeatureBinning<double>> before = {FeatureBinning<double>(2, binning1.begin(), binning1.end()),
                                                  FeatureBinning<double>(2, binning2.begin(), binning2.end())};

    std::stringstream stream;
    stream << before;

    std::vector<FeatureBinning<double>> after;
    stream >> after;

    EXPECT_EQ(before.size(), after.size());
    for(unsigned int j = 0; j < before.size() and j < after.size(); ++j) {

        auto &before_featureBinning = before[j];
        auto &after_featureBinning = after[j];
        const auto &after_binning = after_featureBinning.GetBinning();
        const auto &before_binning = before_featureBinning.GetBinning();

        EXPECT_EQ(before_featureBinning.GetNLevels(), after_featureBinning.GetNLevels());
        EXPECT_EQ(before_binning.size(), after_binning.size());
        for(unsigned int i = 0; i < before_binning.size() and i < after_binning.size(); ++i)
            EXPECT_DOUBLE_EQ(before_binning[i], after_binning[i]);

    }

}

TEST_F(IOTest, IOCut) {

    Cut before;
    before.feature = 1;
    before.gain = 3.4;
    before.index = 5;
    before.valid = true;

    std::stringstream stream;
    stream << before;

    Cut after;
    stream >> after;

    EXPECT_EQ(before.feature, after.feature);
    EXPECT_EQ(before.gain, after.gain);
    EXPECT_EQ(before.index, after.index);
    EXPECT_EQ(before.valid, after.valid);

}
            
TEST_F(IOTest, IOTree) {

    Cut cut1, cut2, cut3;
    cut1.feature = 0;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = -3.0;
    cut2.feature = 1;
    cut2.index = 9;
    cut2.gain = 1.0;
    cut2.valid = true;
    cut3.feature = 0;
    cut3.index = 1;
    cut3.gain = 0.0;
    cut3.valid = false;
    
    std::vector<Cut> before_cuts = {cut1, cut2, cut3};
    std::vector<float> before_purities = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
    std::vector<float> before_boostWeights = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    Tree before(before_cuts, before_purities, before_boostWeights);            
    
    std::stringstream stream;
    stream << before;

    auto after = readTreeFromStream(stream);
    const auto &after_cuts = after.GetCuts();
    const auto &after_purities = after.GetPurities();
    const auto &after_boostWeights = after.GetBoostWeights();

    EXPECT_EQ(before_cuts.size(), after_cuts.size());
    for(unsigned int i = 0; i < before_cuts.size() and i < after_cuts.size(); ++i) {
        EXPECT_DOUBLE_EQ(before_cuts[i].feature, after_cuts[i].feature);
        EXPECT_DOUBLE_EQ(before_cuts[i].valid, after_cuts[i].valid);
        EXPECT_DOUBLE_EQ(before_cuts[i].index, after_cuts[i].index);
        EXPECT_DOUBLE_EQ(before_cuts[i].gain, after_cuts[i].gain);
    }
    
    EXPECT_EQ(before_purities.size(), after_purities.size());
    for(unsigned int i = 0; i < before_purities.size() and i < after_purities.size(); ++i)
        EXPECT_DOUBLE_EQ(before_purities[i], after_purities[i]);
    
    EXPECT_EQ(before_boostWeights.size(), after_boostWeights.size());
    for(unsigned int i = 0; i < before_boostWeights.size() and i < after_boostWeights.size(); ++i)
        EXPECT_DOUBLE_EQ(before_boostWeights[i], after_boostWeights[i]);

}

TEST_F(IOTest, IOForest) {

    Cut cut1, cut2, cut3, cut4;
    cut1.feature = 0;
    cut1.index = 5;
    cut1.valid = true;
    cut1.gain = -3.0;
    cut2.feature = 1;
    cut2.index = 9;
    cut2.gain = 1.0;
    cut2.valid = true;
    cut3.feature = 0;
    cut3.index = 1;
    cut3.gain = 0.0;
    cut3.valid = false;
    cut4.feature = 2;
    cut4.index = 3;
    cut4.valid = true;
    cut4.gain = 1.61;
    
    Forest before(0.5, 1.6);
    before.AddTree(Tree({cut1, cut2, cut3}, { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 }, { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}));
    before.AddTree(Tree({cut1, cut4, cut3}, { 0.6, 0.2, 0.5, 0.4, 0.5, 0.6, 0.7 }, { 2.0, 2.0, 3.0, 5.0, 5.0, 6.0, 1.0}));
    const auto &before_forest = before.GetForest();

    std::stringstream stream;
    stream << before;

    auto after = readForestFromStream(stream);
    const auto &after_forest = after.GetForest();

    EXPECT_EQ(before.GetF0(), after.GetF0());
    EXPECT_EQ(before.GetShrinkage(), after.GetShrinkage());

    EXPECT_EQ(before_forest.size(), after_forest.size());
    for(unsigned int j = 0; j < before_forest.size() and j < after_forest.size(); ++j) {

        auto &before_tree = before_forest[j];
        const auto &before_cuts = before_tree.GetCuts();
        const auto &before_purities = before_tree.GetPurities();
        const auto &before_boostWeights = before_tree.GetBoostWeights();
        
        auto &after_tree = after_forest[j];
        const auto &after_cuts = after_tree.GetCuts();
        const auto &after_purities = after_tree.GetPurities();
        const auto &after_boostWeights = after_tree.GetBoostWeights();

        EXPECT_EQ(before_cuts.size(), after_cuts.size());
        for(unsigned int i = 0; i < before_cuts.size() and i < after_cuts.size(); ++i) {
            EXPECT_DOUBLE_EQ(before_cuts[i].feature, after_cuts[i].feature);
            EXPECT_DOUBLE_EQ(before_cuts[i].valid, after_cuts[i].valid);
            EXPECT_DOUBLE_EQ(before_cuts[i].index, after_cuts[i].index);
            EXPECT_DOUBLE_EQ(before_cuts[i].gain, after_cuts[i].gain);
        }
        
        EXPECT_EQ(before_purities.size(), after_purities.size());
        for(unsigned int i = 0; i < before_purities.size() and i < after_purities.size(); ++i)
            EXPECT_DOUBLE_EQ(before_purities[i], after_purities[i]);
        
        EXPECT_EQ(before_boostWeights.size(), after_boostWeights.size());
        for(unsigned int i = 0; i < before_boostWeights.size() and i < after_boostWeights.size(); ++i)
            EXPECT_DOUBLE_EQ(before_boostWeights[i], after_boostWeights[i]);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
