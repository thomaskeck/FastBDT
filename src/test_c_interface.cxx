/**
 * Thomas Keck 2015
 */

#include "c_interface.h"

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

TEST_F(CInterfaceTest, CreateDefaultValuesAreCorrect) {

    EXPECT_EQ( expertise->nBinningLevels, 8u);
    EXPECT_EQ( expertise->nTrees, 100u);
    EXPECT_EQ( expertise->nLayersPerTree, 3u);
    EXPECT_DOUBLE_EQ( expertise->shrinkage, 0.1);
    EXPECT_DOUBLE_EQ( expertise->randRatio, 0.5);

}

TEST_F(CInterfaceTest, SetNBinningLevelsWorks ) {
    
    SetNBinningLevels(expertise, 10u);
    EXPECT_EQ( expertise->nBinningLevels, 10u);
    SetNBinningLevels(expertise, 5u);
    EXPECT_EQ( expertise->nBinningLevels, 5u);

}

TEST_F(CInterfaceTest, SetNTreesWorks ) {
    
    SetNTrees(expertise, 200u);
    EXPECT_EQ( expertise->nTrees, 200u);
    SetNTrees(expertise, 50u);
    EXPECT_EQ( expertise->nTrees, 50u);

}

TEST_F(CInterfaceTest, SetNLayersPerTreeWorks ) {
    
    SetNLayersPerTree(expertise, 5u);
    EXPECT_EQ( expertise->nLayersPerTree, 5u);
    SetNLayersPerTree(expertise, 2u);
    EXPECT_EQ( expertise->nLayersPerTree, 2u);

}

TEST_F(CInterfaceTest, SetShrinkageWorks ) {
    
    SetShrinkage(expertise, 0.2);
    EXPECT_DOUBLE_EQ( expertise->shrinkage, 0.2);
    SetShrinkage(expertise, 0.4);
    EXPECT_DOUBLE_EQ( expertise->shrinkage, 0.4);

}
    
    
TEST_F(CInterfaceTest, SetRandRatioWorks ) {
    
    SetRandRatio(expertise, 0.6);
    EXPECT_DOUBLE_EQ( expertise->randRatio, 0.6);
    SetRandRatio(expertise, 0.8);
    EXPECT_DOUBLE_EQ( expertise->randRatio, 0.8);

}


TEST_F(CInterfaceTest, TrainAndAnalyseForestWorks ) {

    // Use just one branch instead of a whole forest for testing
    // We only test if the ForestBuilder is called correctly,
    // the builder itself is tested elsewhere.
    SetNTrees(expertise, 10u);
    SetNLayersPerTree(expertise, 1u);
    SetRandRatio(expertise, 1.0);
    SetShrinkage(expertise, 1.0);
    SetNBinningLevels(expertise, 1u);

    double data_ptr[] = {1.0, 2.6, 1.6, 2.5, 1.1, 2.0, 1.9, 2.1, 1.6, 2.9, 1.9, 2.9, 1.5, 2.0};
    unsigned int target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
    Train(expertise, data_ptr, target_ptr, 7, 2);

    EXPECT_EQ(expertise->forest.GetForest().size(), 10u);
    EXPECT_EQ(expertise->forest.GetForest()[0].GetCuts().size(), 1u);
    EXPECT_EQ(expertise->forest.GetForest()[0].GetCuts()[0].feature, 0u);
    EXPECT_EQ(expertise->forest.GetForest()[0].GetCuts()[0].index, 2u);
    EXPECT_TRUE(expertise->forest.GetForest()[0].GetCuts()[0].valid);
    EXPECT_FLOAT_EQ(expertise->forest.GetForest()[0].GetCuts()[0].gain, 1.7142857f);

    double test_ptr[] = {1.0, 2.6};
    EXPECT_LE(Analyse(expertise, test_ptr), 0.001);
}

/* TODO Test this to functions as well using a temporary directory or file
 *
    void Load(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);
      
      std::fstream file(weightfile, std::ios_base::in);
      if(not file)
    	  return;

      file >> expertise->featureBinnings;
      expertise->forest = FastBDT::readForestFromStream(file);
    }

      return expertise->forest.Analyse(bins);
    }

    void Save(void* ptr, char *weightfile) {
      Expertise *expertise = reinterpret_cast<Expertise*>(ptr);

      std::fstream file(weightfile, std::ios_base::out | std::ios_base::trunc);
      file << expertise->featureBinnings << std::endl;
      file << expertise->forest << std::endl;
    }
*/
