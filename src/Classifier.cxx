/*
 * Thomas Keck 2017
 *
 * Simplified sklearn interface
 */


#include "Classifier.h"

namespace FastBDT {

  void Classifier::fit(const std::vector<std::vector<float>> &X, const std::vector<bool> &y, const std::vector<Weight> &w) {

    if(static_cast<int>(X.size()) - static_cast<int>(m_numberOfFlatnessFeatures) <= 0) {
      throw std::runtime_error("FastBDT requires at least one feature");
    }
    m_numberOfFeatures = X.size() - m_numberOfFlatnessFeatures ;
    
    if(m_numberOfFeatures + m_numberOfFlatnessFeatures != m_binning.size()) {
      throw std::runtime_error("Number of features must be equal to the number of provided binnings");
    }
    
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
      
  float Classifier::predict(const std::vector<float> &X) const {

      if(m_can_use_fast_forest) {
        return m_fast_forest.Analyse(X);
      } else {
        std::vector<unsigned int> bins(m_numberOfFinalFeatures);
        unsigned int bin = 0;
        unsigned int pFeature = 0;
        for(unsigned int iFeature = 0; iFeature < m_numberOfFinalFeatures; ++iFeature) {
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
    stream << classifier.m_can_use_fast_forest << std::endl;
    stream << classifier.m_fast_forest << std::endl;
    stream << classifier.m_binned_forest << std::endl;

    return stream;
}

}
