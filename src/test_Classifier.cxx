/**
 * Thomas Keck 2017
 */

#include "Classifier.h"
#include <iostream>
#include <fstream>

#include <gtest/gtest.h>

#include <sstream>
#include <limits>

#include <algorithm>
#include <random>

using namespace FastBDT;


std::vector<std::vector<float>> GetIrisX() {
  std::vector<std::vector<float>> X = {{5.1,4.9,4.7,4.6,5.0,5.4,4.6,5.0,4.4,4.9,5.4,4.8,4.8,4.3,5.8,5.7,5.4,5.1,5.7,5.1,5.4,5.1,4.6,5.1,4.8,5.0,5.0,5.2,5.2,4.7,4.8,5.4,5.2,5.5,4.9,5.0,5.5,4.9,4.4,5.1,5.0,4.5,4.4,5.0,5.1,4.8,5.1,4.6,5.3,5.0,7.0,6.4,6.9,5.5,6.5,5.7,6.3,4.9,6.6,5.2,5.0,5.9,6.0,6.1,5.6,6.7,5.6,5.8,6.2,5.6,5.9,6.1,6.3,6.1,6.4,6.6,6.8,6.7,6.0,5.7,5.5,5.5,5.8,6.0,5.4,6.0,6.7,6.3,5.6,5.5,5.5,6.1,5.8,5.0,5.6,5.7,5.7,6.2,5.1,5.7,6.3,5.8,7.1,6.3,6.5,7.6,4.9,7.3,6.7,7.2,6.5,6.4,6.8,5.7,5.8,6.4,6.5,7.7,7.7,6.0,6.9,5.6,7.7,6.3,6.7,7.2,6.2,6.1,6.4,7.2,7.4,7.9,6.4,6.3,6.1,7.7,6.3,6.4,6.0,6.9,6.7,6.9,5.8,6.8,6.7,6.7,6.3,6.5,6.2,5.9}, {3.5,3.0,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.7,3.4,3.0,3.0,4.0,4.4,3.9,3.5,3.8,3.8,3.4,3.7,3.6,3.3,3.4,3.0,3.4,3.5,3.4,3.2,3.1,3.4,4.1,4.2,3.1,3.2,3.5,3.1,3.0,3.4,3.5,2.3,3.2,3.5,3.8,3.0,3.8,3.2,3.7,3.3,3.2,3.2,3.1,2.3,2.8,2.8,3.3,2.4,2.9,2.7,2.0,3.0,2.2,2.9,2.9,3.1,3.0,2.7,2.2,2.5,3.2,2.8,2.5,2.8,2.9,3.0,2.8,3.0,2.9,2.6,2.4,2.4,2.7,2.7,3.0,3.4,3.1,2.3,3.0,2.5,2.6,3.0,2.6,2.3,2.7,3.0,2.9,2.9,2.5,2.8,3.3,2.7,3.0,2.9,3.0,3.0,2.5,2.9,2.5,3.6,3.2,2.7,3.0,2.5,2.8,3.2,3.0,3.8,2.6,2.2,3.2,2.8,2.8,2.7,3.3,3.2,2.8,3.0,2.8,3.0,2.8,3.8,2.8,2.8,2.6,3.0,3.4,3.1,3.0,3.1,3.1,3.1,2.7,3.2,3.3,3.0,2.5,3.0,3.4,3.0}, {1.4,1.4,1.3,1.5,1.4,1.7,1.4,1.5,1.4,1.5,1.5,1.6,1.4,1.1,1.2,1.5,1.3,1.4,1.7,1.5,1.7,1.5,1.0,1.7,1.9,1.6,1.6,1.5,1.4,1.6,1.6,1.5,1.5,1.4,1.5,1.2,1.3,1.5,1.3,1.5,1.3,1.3,1.3,1.6,1.9,1.4,1.6,1.4,1.5,1.4,4.7,4.5,4.9,4.0,4.6,4.5,4.7,3.3,4.6,3.9,3.5,4.2,4.0,4.7,3.6,4.4,4.5,4.1,4.5,3.9,4.8,4.0,4.9,4.7,4.3,4.4,4.8,5.0,4.5,3.5,3.8,3.7,3.9,5.1,4.5,4.5,4.7,4.4,4.1,4.0,4.4,4.6,4.0,3.3,4.2,4.2,4.2,4.3,3.0,4.1,6.0,5.1,5.9,5.6,5.8,6.6,4.5,6.3,5.8,6.1,5.1,5.3,5.5,5.0,5.1,5.3,5.5,6.7,6.9,5.0,5.7,4.9,6.7,4.9,5.7,6.0,4.8,4.9,5.6,5.8,6.1,6.4,5.6,5.1,5.6,6.1,5.6,5.5,4.8,5.4,5.6,5.1,5.1,5.9,5.7,5.2,5.0,5.2,5.4,5.1}, {0.2,0.2,0.2,0.2,0.2,0.4,0.3,0.2,0.2,0.1,0.2,0.2,0.1,0.1,0.2,0.4,0.4,0.3,0.3,0.3,0.2,0.4,0.2,0.5,0.2,0.2,0.4,0.2,0.2,0.2,0.2,0.4,0.1,0.2,0.1,0.2,0.2,0.1,0.2,0.2,0.3,0.3,0.2,0.6,0.4,0.3,0.2,0.2,0.2,0.2,1.4,1.5,1.5,1.3,1.5,1.3,1.6,1.0,1.3,1.4,1.0,1.5,1.0,1.4,1.3,1.4,1.5,1.0,1.5,1.1,1.8,1.3,1.5,1.2,1.3,1.4,1.4,1.7,1.5,1.0,1.1,1.0,1.2,1.6,1.5,1.6,1.5,1.3,1.3,1.3,1.2,1.4,1.2,1.0,1.3,1.2,1.3,1.3,1.1,1.3,2.5,1.9,2.1,1.8,2.2,2.1,1.7,1.8,1.8,2.5,2.0,1.9,2.1,2.0,2.4,2.3,1.8,2.2,2.3,1.5,2.3,2.0,2.0,1.8,2.1,1.8,1.8,1.8,2.1,1.6,1.9,2.0,2.2,1.5,1.4,2.3,2.4,1.8,1.8,2.1,2.4,2.3,1.9,2.3,2.5,2.3,1.9,2.0,2.3,1.8} };
  return X;
}

std::vector<bool> GetIrisY() {
   std::vector<bool> y = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true};
   return y;
}

std::vector<float> GetIrisW() {
  std::vector<float> w = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
  return w;
}


float GetIrisScore(const Classifier &classifier) {
    auto X = GetIrisX();
    auto y = GetIrisY();
    float sum = 0;
    for(unsigned int i = 0; i < y.size(); ++i) {
      float p = classifier.predict({X[0][i], X[1][i], X[2][i], X[3][i]});
      sum += (y[i]-p)*(y[i]-p);
    }
    return -sum;
}

class ClassifierTest : public ::testing::Test {
    protected:
        virtual void SetUp() {
          X = GetIrisX();
          y = GetIrisY();
          w = GetIrisW();
        }

        virtual void TearDown() {
        }

        std::vector<std::vector<float>> X;
        std::vector<bool> y;
        std::vector<float> w;

};

TEST_F(ClassifierTest, SimpleClassifierWorks) {

    FastBDT::Classifier classifier(10, 3, {4, 4, 4, 4});
    classifier.fit(X, y, w);

    EXPECT_GT(GetIrisScore(classifier), -7.0);
    EXPECT_LT(GetIrisScore(classifier), -5.0);

}

TEST_F(ClassifierTest, MoreTreesAreBetter) {

    FastBDT::Classifier classifier1(1, 1, {4, 4, 4, 4});
    classifier1.fit(X, y, w);
    
    FastBDT::Classifier classifier2(4, 1, {4, 4, 4, 4});
    classifier2.fit(X, y, w);
    
    FastBDT::Classifier classifier3(16, 1, {4, 4, 4, 4});
    classifier3.fit(X, y, w);
    
    FastBDT::Classifier classifier4(64, 1, {4, 4, 4, 4});
    classifier4.fit(X, y, w);

    EXPECT_LT(GetIrisScore(classifier1), GetIrisScore(classifier2));
    EXPECT_LT(GetIrisScore(classifier2), GetIrisScore(classifier3));
    EXPECT_LT(GetIrisScore(classifier3), GetIrisScore(classifier4));

}

TEST_F(ClassifierTest, DeeperTreesAreBetter) {

    FastBDT::Classifier classifier1(1, 1, {4, 4, 4, 4});
    classifier1.fit(X, y, w);
    
    FastBDT::Classifier classifier2(1, 3, {4, 4, 4, 4});
    classifier2.fit(X, y, w);
    
    FastBDT::Classifier classifier3(1, 5, {4, 4, 4, 4});
    classifier3.fit(X, y, w);
    
    FastBDT::Classifier classifier4(1, 7, {4, 4, 4, 4});
    classifier4.fit(X, y, w);

    EXPECT_LT(GetIrisScore(classifier1), GetIrisScore(classifier2));
    EXPECT_LT(GetIrisScore(classifier2), GetIrisScore(classifier3));
    EXPECT_LT(GetIrisScore(classifier3), GetIrisScore(classifier4));

}


TEST_F(ClassifierTest, MoreBinsAreBetter) {

    FastBDT::Classifier classifier1(1, 3, {2, 2, 2, 2});
    classifier1.fit(X, y, w);
    
    FastBDT::Classifier classifier2(1, 3, {2, 3, 2, 3});
    classifier2.fit(X, y, w);
    
    FastBDT::Classifier classifier3(1, 3, {3, 4, 3, 4});
    classifier3.fit(X, y, w);
    
    FastBDT::Classifier classifier4(1, 3, {4, 5, 4, 5});
    classifier4.fit(X, y, w);

    EXPECT_LT(GetIrisScore(classifier1), GetIrisScore(classifier2));
    EXPECT_LT(GetIrisScore(classifier2), GetIrisScore(classifier3));
    EXPECT_LT(GetIrisScore(classifier3), GetIrisScore(classifier4));

}

TEST_F(ClassifierTest, HeavilyOvertrainedBDTIsPerfect) {

    FastBDT::Classifier classifier(100, 10, {8, 8, 8, 8});
    classifier.fit(X, y, w);

    EXPECT_GT(GetIrisScore(classifier), -0.01f);

}

TEST_F(ClassifierTest, SubsamplingChangesResult) {

    FastBDT::Classifier classifier1(1, 5, {4, 4, 4, 4}, 0.1, 0.5);
    classifier1.fit(X, y, w);
    
    FastBDT::Classifier classifier2(1, 5, {4, 4, 4, 4}, 0.1, 0.5);
    classifier2.fit(X, y, w);

    EXPECT_NE(GetIrisScore(classifier1), GetIrisScore(classifier2));

}

TEST_F(ClassifierTest, LoadAndSaveWorks) {

    FastBDT::Classifier classifier(10, 3, {4, 4, 4, 4});
    classifier.fit(X, y, w);
    
    float score1 = GetIrisScore(classifier);

    std::fstream file_out("unittest.weightfile", std::ios_base::out | std::ios_base::trunc);
    file_out << classifier << std::endl;
    file_out.close();

    std::fstream file_in("unittest.weightfile", std::ios_base::in);
    FastBDT::Classifier classifier2(file_in);
    file_in.close();
    
    float score2 = GetIrisScore(classifier2);

    EXPECT_FLOAT_EQ(score1, score2);
}

