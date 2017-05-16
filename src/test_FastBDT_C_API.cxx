/**
 * Thomas Keck 2015
 */

#include "FastBDT_C_API.h"

#include <gtest/gtest.h>

class CInterfaceTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
            expertise = static_cast<Expertise*>(Create());
        }

        virtual void TearDown() {
            Delete(expertise);
        }

        Expertise *expertise;

};

TEST_F(CInterfaceTest, SetGetBinning ) {

    unsigned int binning[] = {10u, 20u};
    SetBinning(expertise, binning, 2);
    EXPECT_EQ(expertise->classifier.GetBinning().size(), 2u);
    EXPECT_EQ(expertise->classifier.GetBinning()[0], 10u);
    EXPECT_EQ(expertise->classifier.GetBinning()[1], 20u);

}

TEST_F(CInterfaceTest, SetGetPurityTransformation ) {
    
    bool purityTransformation[] = {true, false};
    SetPurityTransformation(expertise, purityTransformation, 2);
    EXPECT_EQ(expertise->classifier.GetPurityTransformation().size(), 2u);
    EXPECT_EQ(expertise->classifier.GetPurityTransformation()[0], true);
    EXPECT_EQ(expertise->classifier.GetPurityTransformation()[1], false);

}

TEST_F(CInterfaceTest, SetGetNTrees ) {
    
    SetNTrees(expertise, 200u);
    EXPECT_EQ(expertise->classifier.GetNTrees(), 200u);

}

TEST_F(CInterfaceTest, SetGetSPlot ) {
    
    SetSPlot(expertise, false);
    EXPECT_EQ(expertise->classifier.GetSPlot(), false);
    SetSPlot(expertise, true);
    EXPECT_EQ(expertise->classifier.GetSPlot(), true);

}

TEST_F(CInterfaceTest, SetGetTransform2Probability ) {
    
    SetTransform2Probability(expertise, false);
    EXPECT_EQ(expertise->classifier.GetTransform2Probability(), false);
    SetTransform2Probability(expertise, true);
    EXPECT_EQ(expertise->classifier.GetTransform2Probability(), true);

}

TEST_F(CInterfaceTest, SetGetDepth ) {
    
    SetDepth(expertise, 5u);
    EXPECT_EQ(expertise->classifier.GetDepth(), 5u);
    SetDepth(expertise, 2u);
    EXPECT_EQ(expertise->classifier.GetDepth(), 2u);

}

TEST_F(CInterfaceTest, SetGetFlatnessLossWorks ) {
    
    SetFlatnessLoss(expertise, 0.2);
    EXPECT_DOUBLE_EQ(expertise->classifier.GetFlatnessLoss(), 0.2);
    SetFlatnessLoss(expertise, 0.4);
    EXPECT_DOUBLE_EQ(expertise->classifier.GetFlatnessLoss(), 0.4);

}

TEST_F(CInterfaceTest, SetGetShrinkageWorks ) {
    
    SetShrinkage(expertise, 0.2);
    EXPECT_DOUBLE_EQ(expertise->classifier.GetShrinkage(), 0.2);
    SetShrinkage(expertise, 0.4);
    EXPECT_DOUBLE_EQ(expertise->classifier.GetShrinkage(), 0.4);

}
    
    
TEST_F(CInterfaceTest, SetSubsampleWorks ) {
    
    SetSubsample(expertise, 0.6);
    EXPECT_DOUBLE_EQ(expertise->classifier.GetSubsample(), 0.6);
    SetSubsample(expertise, 0.8);
    EXPECT_DOUBLE_EQ(expertise->classifier.GetSubsample(), 0.8);

}


TEST_F(CInterfaceTest, FitAndPredictWorksWithoutWeights ) {

    // Use just one branch instead of a whole forest for testing
    // We only test if the ForestBuilder is called correctly,
    // the builder itself is tested elsewhere.
    SetNTrees(expertise, 10u);
    SetDepth(expertise, 1u);
    SetSubsample(expertise, 1.0);
    SetShrinkage(expertise, 1.0);
    unsigned int binning[] = {2u, 2u};
    SetBinning(expertise, binning, 2);
    SetTransform2Probability(expertise, true);
    SetNumberOfFlatnessFeatures(expertise, 0);

    float data_ptr[] = {1.0, 2.6, 1.6, 2.5, 1.1, 2.0, 1.9, 2.1, 1.6, 2.9, 1.9, 2.9, 1.5, 2.0};
    bool target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
    Fit(expertise, data_ptr, nullptr, target_ptr, 7, 2);

    float test_ptr[] = {1.0, 2.6};
    EXPECT_LE(Predict(expertise, test_ptr), 0.01);
    
    float test_ptr2[] = {1.6, 2.5};
    EXPECT_GE(Predict(expertise, test_ptr2), 0.99);
}


TEST_F(CInterfaceTest, TrainAndAnalyseForestWorksWithSpectators ) {

    // Use just one branch instead of a whole forest for testing
    // We only test if the ForestBuilder is called correctly,
    // the builder itself is tested elsewhere.
    SetNTrees(expertise, 10u);
    SetDepth(expertise, 1u);
    SetSubsample(expertise, 1.0);
    SetShrinkage(expertise, 1.0);
    unsigned int binning[] = {2u, 2u, 2u, 3u};
    SetBinning(expertise, binning, 4);
    SetTransform2Probability(expertise, true);
    SetNumberOfFlatnessFeatures(expertise, 2);

    float data_ptr[] = {1.0, 2.6, 0.0, -10.0, 
                         1.6, 2.5, 99.0, 0.0,
                         1.1, 2.0, -500.0, 12.1,
                         1.9, 2.1, 0.0, 0.0,
                         1.6, 2.9, 23.0, 42.0,
                         1.9, 2.9, 0.0, 1.0,
                         1.5, 2.0, 1.0, -1.0};
    bool target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
    Fit(expertise, data_ptr, nullptr, target_ptr, 7, 4);

    float test_ptr[] = {1.0, 2.6};
    EXPECT_LE(Predict(expertise, test_ptr), 0.03);
}

TEST_F(CInterfaceTest, TrainAndAnalyseForestWorksWithWeights ) {

    // Use just one branch instead of a whole forest for testing
    // We only test if the ForestBuilder is called correctly,
    // the builder itself is tested elsewhere.
    SetNTrees(expertise, 10u);
    SetDepth(expertise, 1u);
    SetSubsample(expertise, 1.0);
    SetShrinkage(expertise, 1.0);
    unsigned int binning[] = {2u, 2u};
    SetBinning(expertise, binning, 2);
    SetTransform2Probability(expertise, true);
    SetNumberOfFlatnessFeatures(expertise, 0);

    float data_ptr[] = {1.0, 2.6, 1.6, 2.5, 1.1, 2.0, 1.9, 2.1, 1.6, 2.9, 1.9, 2.9, 1.5, 2.0};
    float weight_ptr[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    bool target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
    Fit(expertise, data_ptr, weight_ptr, target_ptr, 7, 2);

    float test_ptr[] = {1.0, 2.6};
    EXPECT_LE(Predict(expertise, test_ptr), 0.01);
    
    float weight_ptr2[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    Fit(expertise, data_ptr, weight_ptr2, target_ptr, 7, 2);
    EXPECT_LE(Predict(expertise, test_ptr), 0.01);
    
    float weight_ptr3[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
    Fit(expertise, data_ptr, weight_ptr3, target_ptr, 7, 2);
    EXPECT_LE(Predict(expertise, test_ptr), 0.03);
}
