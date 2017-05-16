/*
 * Thomas Keck 2017
 *
 * Simplified sklearn interface
 */


#include "Classifier.h"
#include <iostream>

namespace FastBDT {

  void Classifier::fit(const std::vector<std::vector<float>> &X, const std::vector<bool> &y, const std::vector<Weight> &w) {

    if(static_cast<int>(X.size()) - static_cast<int>(m_numberOfFlatnessFeatures) <= 0) {
      throw std::runtime_error("FastBDT requires at least one feature");
    }
    m_numberOfFeatures = X.size() - m_numberOfFlatnessFeatures ;

    if(m_binning.size() == 0) {
      for(unsigned int i = 0; i < X.size(); ++i)
        m_binning.push_back(8);
    }

    if(m_numberOfFeatures + m_numberOfFlatnessFeatures != m_binning.size()) {
      throw std::runtime_error("Number of features must be equal to the number of provided binnings");
    }
    
    if(m_purityTransformation.size() == 0) {
      for(unsigned int i = 0; i < m_binning.size() - m_numberOfFlatnessFeatures; ++i)
        m_purityTransformation.push_back(false);
    }

    for(auto p : m_purityTransformation)
      if(p)
        m_can_use_fast_forest = false;
    
    if(m_numberOfFeatures != m_purityTransformation.size()) {
      throw std::runtime_error("Number of ordinary features must be equal to the number of provided purityTransformation flags.");
    }

    unsigned int numberOfEvents = X[0].size();
    if(numberOfEvents == 0) {
      throw std::runtime_error("FastBDT requires at least one event");
    }

    if(numberOfEvents != y.size()) {
      throw std::runtime_error("Number of data-points X doesn't match the numbers of labels y");
    }
    
    if(numberOfEvents != w.size()) {
      throw std::runtime_error("Number of data-points X doesn't match the numbers of weights w");
    }

    m_numberOfFinalFeatures = m_numberOfFeatures;
    for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
      auto feature = X[iFeature];
      m_featureBinning.push_back(FeatureBinning<float>(m_binning[iFeature], feature));
      if(m_purityTransformation[iFeature]) {
        m_numberOfFinalFeatures++;
        std::vector<unsigned int> feature(numberOfEvents);
        for(unsigned int iEvent = 0; iEvent < numberOfEvents; ++iEvent) {
          feature[iEvent] = m_featureBinning[iFeature].ValueToBin(X[iFeature][iEvent]);
        }
        m_purityBinning.push_back(PurityTransformation(m_binning[iFeature], feature, w, y));
        m_binning.insert(m_binning.begin() + iFeature + 1, m_binning[iFeature]);
      }
    }
    
    for(unsigned int iFeature = 0; iFeature < m_numberOfFlatnessFeatures; ++iFeature) {
      auto feature = X[iFeature + m_numberOfFeatures];
      m_featureBinning.push_back(FeatureBinning<float>(m_binning[iFeature + m_numberOfFinalFeatures], feature));
    }
  
    EventSample eventSample(numberOfEvents, m_numberOfFinalFeatures, m_numberOfFlatnessFeatures, m_binning);
    std::vector<unsigned int> bins(m_numberOfFinalFeatures+m_numberOfFlatnessFeatures);

    for(unsigned int iEvent = 0; iEvent < numberOfEvents; ++iEvent) {
      unsigned int bin = 0;
      unsigned int pFeature = 0; 
      for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
        bins[bin] = m_featureBinning[iFeature].ValueToBin(X[iFeature][iEvent]);
        bin++;
        if(m_purityTransformation[iFeature]) {
          bins[bin] = m_purityBinning[pFeature].BinToPurityBin(bins[bin-1]);
          pFeature++;
          bin++;
        }
      }
      for(unsigned int iFeature = 0; iFeature < m_numberOfFlatnessFeatures; ++iFeature) {
        bins[bin] = m_featureBinning[iFeature + m_numberOfFeatures].ValueToBin(X[iFeature + m_numberOfFeatures][iEvent]);
        bin++;
      }
      eventSample.AddEvent(bins, w[iEvent], y[iEvent] == 1);
    }
   
    m_featureBinning.resize(m_numberOfFeatures);

    ForestBuilder df(eventSample, m_nTrees, m_shrinkage, m_subsample, m_depth, m_sPlot, m_flatnessLoss);
    if(m_can_use_fast_forest) {
        Forest<float> temp_forest( df.GetShrinkage(), df.GetF0(), m_transform2probability);
        for( auto t : df.GetForest() ) {
           temp_forest.AddTree(removeFeatureBinningTransformationFromTree(t, m_featureBinning));
        }
        m_fast_forest = temp_forest;
    } else {
        Forest<unsigned int> temp_forest(df.GetShrinkage(), df.GetF0(), m_transform2probability);
        for( auto t : df.GetForest() ) {
           temp_forest.AddTree(t);
        }
        m_binned_forest = temp_forest;
    }

  }

  void Classifier::Print() {

    std::cout << "NTrees " << m_nTrees << std::endl;
    std::cout << "Depth " << m_depth << std::endl;
    std::cout << "NumberOfFeatures " << m_numberOfFeatures << std::endl;

  }
      
  float Classifier::predict(const std::vector<float> &X) const {

      if(m_can_use_fast_forest) {
        return m_fast_forest.Analyse(X);
      } else {
        std::vector<unsigned int> bins(m_numberOfFinalFeatures);
        unsigned int bin = 0;
        unsigned int pFeature = 0;
        for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
          bins[bin] = m_featureBinning[iFeature].ValueToBin(X[iFeature]);
          bin++;
          if(m_purityTransformation[iFeature]) {
              bins[bin] = m_purityBinning[pFeature].BinToPurityBin(bins[bin-1]);
              pFeature++;
              bin++;
          }
        }
        return m_binned_forest.Analyse(bins);
      }
  }
  
  std::map<unsigned int, double> Classifier::GetIndividualVariableRanking(const std::vector<float> &X) const {
    
      std::map<unsigned int, double> ranking;

      if(m_can_use_fast_forest) {
        ranking = m_fast_forest.GetIndividualVariableRanking(X);
      } else {
        std::vector<unsigned int> bins(m_numberOfFinalFeatures);
        unsigned int bin = 0;
        unsigned int pFeature = 0;
        for(unsigned int iFeature = 0; iFeature < m_numberOfFeatures; ++iFeature) {
          bins[bin] = m_featureBinning[iFeature].ValueToBin(X[iFeature]);
          bin++;
          if(m_purityTransformation[iFeature]) {
              bins[bin] = m_purityBinning[pFeature].BinToPurityBin(bins[bin-1]);
              pFeature++;
              bin++;
          }
        }
        ranking = m_binned_forest.GetIndividualVariableRanking(bins);
      }

      return MapRankingToOriginalFeatures(ranking);
  }

  std::map<unsigned int, unsigned int> Classifier::GetFeatureMapping() const {
    
    std::map<unsigned int, unsigned int> transformed2original;
    unsigned int transformedFeature = 0;
    for(unsigned int originalFeature = 0; originalFeature < m_numberOfFeatures; ++originalFeature) {
      transformed2original[transformedFeature] = originalFeature;
      if(m_purityTransformation[originalFeature]) {
        transformedFeature++;
        transformed2original[transformedFeature] = originalFeature;
      }
      transformedFeature++;
    }

    return transformed2original;

  }

  std::map<unsigned int, double> Classifier::MapRankingToOriginalFeatures(std::map<unsigned int, double> ranking) const {
    auto transformed2original = GetFeatureMapping();
    std::map<unsigned int, double> original_ranking;
    for(auto &pair : ranking) {
      if(original_ranking.find(transformed2original[pair.first]) == original_ranking.end())
        original_ranking[transformed2original[pair.first]] = 0;
      original_ranking[transformed2original[pair.first]] += pair.second;
    }
    return original_ranking;
  }


  std::map<unsigned int, double> Classifier::GetVariableRanking() const {
    std::map<unsigned int, double> ranking;
    if (m_can_use_fast_forest)
      ranking = m_fast_forest.GetVariableRanking();
    else
      ranking = m_binned_forest.GetVariableRanking();
    return MapRankingToOriginalFeatures(ranking);
  }


std::ostream& operator<<(std::ostream& stream, const Classifier& classifier) {

    stream << classifier.m_version << std::endl;
    stream << classifier.m_nTrees << std::endl;
    stream << classifier.m_depth << std::endl;
    stream << classifier.m_binning << std::endl;
    stream << classifier.m_shrinkage << std::endl;
    stream << classifier.m_subsample << std::endl;
    stream << classifier.m_sPlot << std::endl;
    stream << classifier.m_flatnessLoss << std::endl;
    stream << classifier.m_purityTransformation << std::endl;
    stream << classifier.m_transform2probability << std::endl;
    stream << classifier.m_featureBinning << std::endl;
    stream << classifier.m_purityBinning << std::endl;
    stream << classifier.m_numberOfFeatures << std::endl;
    stream << classifier.m_numberOfFinalFeatures << std::endl;
    stream << classifier.m_numberOfFlatnessFeatures << std::endl;
    stream << classifier.m_can_use_fast_forest << std::endl;
    stream << classifier.m_fast_forest << std::endl;
    stream << classifier.m_binned_forest << std::endl;

    return stream;
}

}
