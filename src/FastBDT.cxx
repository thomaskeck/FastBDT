/**
 * Thomas Keck 2014
 */

#include "FastBDT.h"
#include "FastBDT_IO.h"

#include <iostream>
#include <algorithm>

namespace FastBDT {

  std::vector<Weight> EventWeights::GetSums(unsigned int nSignals) const {

    // Vectorizing FTW!
    std::vector<Weight> sums(3,0);
    for(unsigned int i = 0; i < nSignals; ++i) {
      sums[0] += boost_weights[i] * original_weights[i];
      sums[2] += boost_weights[i]*boost_weights[i] * original_weights[i];
    }

    for(unsigned int i = nSignals; i < original_weights.size(); ++i) {
      sums[1] += boost_weights[i] * original_weights[i];
      sums[2] += boost_weights[i]*boost_weights[i] * original_weights[i];
    }
    return sums;

  }
  
  EventValues::EventValues(unsigned int nEvents, unsigned int nFeatures, unsigned int nSpectators, const std::vector<unsigned int> &nLevels) : values(nEvents*(nFeatures+nSpectators), 0), nFeatures(nFeatures), nSpectators(nSpectators) {

    if(nFeatures + nSpectators != nLevels.size()) {
      throw std::runtime_error("Number of features must be the same as the number of provided binning levels! " + std::to_string(nFeatures) + " + " + std::to_string(nSpectators) + " vs " + std::to_string(nLevels.size()));
    }

    nBins.reserve(nLevels.size());
    for(auto& nLevel : nLevels) 
      nBins.push_back((1 << nLevel)+1);
    
    nBinSums.reserve(nLevels.size()+1);
    nBinSums.push_back(0);
    for(auto &nBin : nBins) 
      nBinSums.push_back(nBinSums.back() + nBin);

  }

  void EventValues::Set(unsigned int iEvent, const std::vector<unsigned int> &features) {

    // Check if the feature vector has the correct size
    if(features.size() != nFeatures + nSpectators) {
      throw std::runtime_error(std::string("Promised number of features are not provided. ") + std::to_string(features.size()) + " vs " + std::to_string(nFeatures) + " + " + std::to_string(nSpectators));
    }

    // Check if the feature values are in the correct range
    for(unsigned int iFeature = 0; iFeature < nFeatures+nSpectators; ++iFeature) {
      if( features[iFeature] > nBins[iFeature] )
        throw std::runtime_error(std::string("Promised number of bins is violated. ") + std::to_string(features[iFeature]) + " vs " + std::to_string(nBins[iFeature]));
    }

    // Now add the new values to the values vector.
    for(unsigned int iFeature = 0; iFeature < nFeatures+nSpectators; ++iFeature) {
      values[iEvent*(nFeatures+nSpectators) + iFeature] = features[iFeature];
    }

  }

  void EventSample::AddEvent(const std::vector<unsigned int> &features, Weight weight, bool isSignal) {

    // First check of we have enough space for an additional event. As the number of
    // events is fixed in the constructor (to avoid time consuming reallocations)
    if(nSignals + nBckgrds == nEvents) {
      throw std::runtime_error(std::string("Promised maximum number of events exceeded. ") + std::to_string(nSignals) + " + " + std::to_string(nBckgrds) + " vs " + std::to_string(nEvents) );
    }
    
    if(std::isnan(weight)) {
      throw std::runtime_error("NAN values as weights are not supported!");
    }

    // Now add the weight and the features at the right position of the arrays.
    // To do so, we calculate the correct index of this event. If it's a signal
    // event we store it right after the last signal event, starting at the 0 position.
    // If it's a background event, we store it right before the last added background event,
    // starting at the nEvents-1 position. We also update the weight sums and amount counts.
    unsigned int index = 0;
    if( isSignal ) {
      index = nSignals;
      ++nSignals;
    } else {
      index = nEvents - 1 - nBckgrds;
      ++nBckgrds;
    }
    weights.SetOriginalWeight(index, weight);
    values.Set(index, features);

  }

  Weight LossFunction(const Weight &nSignal, const Weight &nBckgrd) {
    // Gini-Index x total number of events (needed to calculate information gain efficiently)!
    if( nSignal <= 0 or nBckgrd <= 0 )
      return 0; 
    return (nSignal*nBckgrd)/(nSignal+nBckgrd);
    //return (nSignal*nBckgrd)/((nSignal+nBckgrd)*(nSignal+nBckgrd));
  }

  CumulativeDistributions::CumulativeDistributions(const unsigned int iLayer, const EventSample &sample) {

    const auto &values = sample.GetValues();
    nFeatures = values.GetNFeatures();
    nNodes = (1 << iLayer);
    nBins = values.GetNBins();
    nBinSums = values.GetNBinSums();

    signalCDFs = CalculateCDFs(sample, 0, sample.GetNSignals());
    bckgrdCDFs = CalculateCDFs(sample, sample.GetNSignals(), sample.GetNEvents());

  }

  std::vector<Weight> CumulativeDistributions::CalculateCDFs(const EventSample &sample, const unsigned int firstEvent, const unsigned int lastEvent) const {

    const auto &values = sample.GetValues();
    const auto &flags = sample.GetFlags();
    const auto &weights = sample.GetWeights();

    std::vector<Weight> bins( nNodes*nBinSums[nFeatures] );

    // Fill Cut-PDFs for all nodes in this layer and for every feature
    for(unsigned int iEvent = firstEvent; iEvent < lastEvent; ++iEvent) {
      if( flags.Get(iEvent) < static_cast<int>(nNodes) )
        continue;
      const unsigned int index = (flags.Get(iEvent)-nNodes)*nBinSums[nFeatures];
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature ) {
        const unsigned int subindex = nBinSums[iFeature] + values.Get(iEvent,iFeature);
        bins[index+subindex] += weights.GetOriginalWeight(iEvent) * (weights.GetBoostWeight(iEvent) + weights.GetFlatnessWeight(iEvent));
      }
    }

    // Sum up Cut-PDFs to culumative Cut-PDFs
    for(unsigned int iNode = 0; iNode < nNodes; ++iNode) {
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
        // Start at 2, this ignore the NaN bin at 0!
        for(unsigned int iBin = 2; iBin < nBins[iFeature]; ++iBin) {
          unsigned int index = iNode*nBinSums[nFeatures] + nBinSums[iFeature] + iBin;
          bins[index] += bins[index-1];
        }
      }
    }

    return bins;
  }

  Cut<unsigned int> Node::CalculateBestCut(const CumulativeDistributions &CDFs) const {

    Cut<unsigned int> cut;

    const unsigned int nFeatures = CDFs.GetNFeatures();
    const auto& nBins = CDFs.GetNBins();

    Weight currentLoss = LossFunction(signal, bckgrd);
    if( currentLoss == 0 )
      return cut;

    // Loop over all features and cuts and sum up signal and background histograms to cumulative histograms
    for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
      // Start at 2, this ignores the NaN bin at 0
      for(unsigned int iCut = 2; iCut < nBins[iFeature]; ++iCut) {
        Weight s = CDFs.GetSignal(iNode, iFeature, iCut-1);
        Weight b = CDFs.GetBckgrd(iNode, iFeature, iCut-1);
        Weight currentGain = currentLoss - LossFunction( signal-s, bckgrd-b ) - LossFunction( s, b );

        if( cut.gain <= currentGain ) {
          cut.gain = currentGain;
          cut.feature = iFeature;
          cut.index = iCut;
          cut.valid = true;
        }
      }
    }

    return cut;

  }

  void Node::AddSignalWeight(Weight weight, Weight original_weight) {
    if(original_weight == 0)
      return;
    signal += weight * original_weight;
    square += weight*weight * original_weight;
  }


  void Node::AddBckgrdWeight(Weight weight, Weight original_weight) {
    if(original_weight == 0)
      return;
    bckgrd += weight * original_weight;
    square += weight*weight * original_weight;
  }

  void Node::SetWeights(std::vector<Weight> weights) {
    signal = weights[0];
    bckgrd = weights[1];
    square = weights[2];
  }

  Weight Node::GetBoostWeight() const {

    Weight denominator = (2*(signal+bckgrd)-square);
    if( denominator == 0 ) {
        if(signal == bckgrd)
            return 0;
        if(signal > bckgrd)
            return 999.0;
        else
            return -999.0;
    }
    Weight value = (signal - bckgrd)/denominator;
    if( value > 999.0 or value < -999.0 ) {
        if(signal > bckgrd)
            return 999.0;
        else
            return -999.0;
    }
    return value;

  }

  void Node::Print() const {
    std::cout << "Node: " << iNode << std::endl;
    std::cout << "Layer: " << iLayer << std::endl;
    std::cout << "Signal: " << signal << std::endl;
    std::cout << "Bckgrd: " << bckgrd << std::endl;
    std::cout << "Square: " << square << std::endl;
  }
  
  /**
   * In bin-space NaN is marked by bin 0
   */
  template<>
  bool is_nan(const unsigned int &value) {
    return value == 0;
  }


  TreeBuilder::TreeBuilder(unsigned int nLayers, EventSample &sample) : nLayers(nLayers) {

    const unsigned int nNodes = 1 << nLayers;
    cuts.resize(nNodes - 1);

    for(unsigned int iLayer = 0; iLayer <= nLayers; ++iLayer) {
      for(unsigned int iNode = 0; iNode < static_cast<unsigned int>(1<<iLayer); ++iNode) {
        nodes.push_back( Node(iLayer, iNode) );
      }
    }

    // The flag of every event is used for two things:
    // Firstly, a flag > 0, determines the node which holds this event at the moment
    // the trees are enumerated from top to bottom from left to right, starting at 1.
    // Secondly, a flag <= 0, disables this event, so it isn't used.
    // flag == 0 means disabled by stochastic bagging
    // flag < 0 means disabled due to missing value, where -flag is the node the event belongs to
    // Initially all events which are not disabled get the flag 1.
    //
    // All the flags of the enabled events are set to 1 by the DecisionForest
    // prepareEventSample method. So there's no need to do this here again.

    // The number of signal and bckgrd events at the root node, is given by the total
    // number of signal and background in the sample.
    const auto sums = sample.GetWeights().GetSums(sample.GetNSignals());
    nodes[0].SetWeights(sums);

    // The training of the tree is done level by level. So we iterate over the levels of the tree
    // and create histograms for signal and background events for different cuts, nodes and features.
    for(unsigned int iLayer = 0; iLayer < nLayers; ++iLayer) {

      CumulativeDistributions CDFs(iLayer, sample);
      UpdateCuts(CDFs, iLayer);
      UpdateFlags(sample);
      UpdateEvents(sample, iLayer);   

    } 

  }

  void TreeBuilder::UpdateCuts(const CumulativeDistributions &CDFs, unsigned int iLayer) {

    for(auto &node : nodes) {
      if( node.IsInLayer(iLayer) ) {
        cuts[ node.GetPosition() ] = node.CalculateBestCut(CDFs);
      }
    }
  }

  void TreeBuilder::UpdateFlags(EventSample &sample) {

    auto &flags = sample.GetFlags();
    const auto &values = sample.GetValues();
    // Iterate over all signal events, and update weights in each node of the next level according to the cuts.
    for(unsigned int iEvent = 0; iEvent < sample.GetNEvents(); ++iEvent) {

      const int flag = flags.Get(iEvent);
      if( flag <= 0)
        continue;
      auto &cut = cuts[flag-1];
      if( not cut.valid )
        continue;

      const unsigned int index = values.Get(iEvent, cut.feature );
      // If NaN value we throw out the event, but remeber its current node using the a negative flag!
      if( index == 0 ) {
        flags.Set(iEvent, -flag);
      } else if( index < cut.index ) {
        flags.Set(iEvent, flag * 2);
      } else {
        flags.Set(iEvent, flag * 2 + 1);
      }
    }
  }

  void TreeBuilder::UpdateEvents(const EventSample &sample, unsigned int iLayer) {

    const unsigned int nNodes = (1 << iLayer);
    const auto &weights = sample.GetWeights();
    const auto &flags = sample.GetFlags();

    for(unsigned int iEvent = 0; iEvent < sample.GetNSignals(); ++iEvent) {
      const int flag = flags.Get(iEvent);
      if( flag >= static_cast<int>(nNodes) ) {
        nodes[flag-1].AddSignalWeight( weights.GetBoostWeight(iEvent), weights.GetOriginalWeight(iEvent) );
      }
    }
    for(unsigned int iEvent = sample.GetNSignals(); iEvent < sample.GetNEvents(); ++iEvent) {
      const int flag = flags.Get(iEvent);
      if( flag >= static_cast<int>(nNodes) ) {
        nodes[flag-1].AddBckgrdWeight( weights.GetBoostWeight(iEvent), weights.GetOriginalWeight(iEvent) );
      }
    }

  }


  void TreeBuilder::Print() const {

    std::cout << "Start Printing Tree" << std::endl;

    for(auto &node : nodes) {
      node.Print();
      std::cout << std::endl;
    }

    for(auto &cut : cuts) {
      std::cout << "Index: " << cut.index << std::endl;
      std::cout << "Feature: " << cut.feature << std::endl;
      std::cout << "Gain: " << cut.gain << std::endl;
      std::cout << "Valid: " << cut.valid << std::endl;
      std::cout << std::endl;
    }

    std::cout << "Finished Printing Tree" << std::endl;
  }

  ForestBuilder::ForestBuilder(EventSample &sample, unsigned int nTrees, double shrinkage, double randRatio, unsigned int nLayersPerTree, bool sPlot, double flatnessLoss) : shrinkage(shrinkage), flatnessLoss(flatnessLoss) {

    auto &weights = sample.GetWeights();
    sums = weights.GetSums(sample.GetNSignals()); 
    // Calculating the initial F value from the proportion of the number of signal and background events in the sample
    double average = (sums[0] - sums[1])/(sums[0] + sums[1]);
    F0 = 0.5*std::log((1+average)/(1-average));
    
    // Apply F0 to original_weights because F0 is not a boost_weight, otherwise prior probability in case of
    // Events with missing values is wrong.
    if (F0 != 0.0) {
        const unsigned int nEvents = sample.GetNEvents();
        const unsigned int nSignals = sample.GetNSignals();
        for(unsigned int iEvent = 0; iEvent < nSignals; ++iEvent)
          weights.SetOriginalWeight(iEvent, 2.0 * sums[1] / (sums[0] + sums[1]) * weights.GetOriginalWeight(iEvent));
        for(unsigned int iEvent = nSignals; iEvent < nEvents; ++iEvent)
          weights.SetOriginalWeight(iEvent, 2.0 * sums[0] / (sums[0] + sums[1]) * weights.GetOriginalWeight(iEvent));
    }
        
    // Resize the FCache to the number of events, and initalise it with the inital 0.0 value
    // Not F0 because F0 is already used in the original_weights
    FCache.resize(sample.GetNEvents(), 0.0);
     
    // Reserve enough space for the boost_weights and trees, to avoid reallocations
    forest.reserve(nTrees);
     
    // Reserve enough space for binned uniform spectators
    if(flatnessLoss > 0) {
        const auto &values = sample.GetValues();
        auto nFeatures = values.GetNFeatures();
        auto nSpectators = values.GetNSpectators();
        auto &nBins = values.GetNBins();
        const unsigned int nEvents = sample.GetNEvents();
        const unsigned int nSignals = sample.GetNSignals();

        signal_event_index_sorted_by_F.resize(nSignals);
        bckgrd_event_index_sorted_by_F.resize(nEvents - nSignals);

        uniform_bin_weight_signal.resize(nSpectators);
        uniform_bin_weight_bckgrd.resize(nSpectators);
        weight_below_current_F_per_uniform_bin.resize(nSpectators);
        for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
          uniform_bin_weight_signal[iSpectator].resize(nBins[nFeatures + iSpectator], 0.0);
          uniform_bin_weight_bckgrd[iSpectator].resize(nBins[nFeatures + iSpectator], 0.0);
          weight_below_current_F_per_uniform_bin[iSpectator].resize(nBins[nFeatures + iSpectator], 0.0);
        }
          
        for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
          for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
            const uint64_t uniformBin = values.GetSpectator(iEvent, iSpectator);
            if (iEvent < nSignals)
              uniform_bin_weight_signal[iSpectator][uniformBin] += weights.GetOriginalWeight(iEvent);
            else
              uniform_bin_weight_bckgrd[iSpectator][uniformBin] += weights.GetOriginalWeight(iEvent);
          }
        }
        for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
          for(uint64_t iUniformBin = 0; iUniformBin < uniform_bin_weight_signal[iSpectator].size(); ++iUniformBin) {
              uniform_bin_weight_signal[iSpectator][iUniformBin] /= sums[0];
          }
          for(uint64_t iUniformBin = 0; iUniformBin < uniform_bin_weight_bckgrd[iSpectator].size(); ++iUniformBin) {
              uniform_bin_weight_bckgrd[iSpectator][iUniformBin] /= sums[1];
          }
        }
    }

    // Now train config.nTrees!
    for(unsigned int iTree = 0; iTree < nTrees; ++iTree) {

      // Update the event weights according to their F value
      updateEventWeights(sample);

      // Add flatness loss terms
      if(flatnessLoss > 0 and iTree > 0) 
          updateEventWeightsWithFlatnessPenalty(sample);

      // Prepare the flags of the events
      prepareEventSample( sample, randRatio, sPlot );   

      // Create and train a new train on the sample
      TreeBuilder builder(nLayersPerTree, sample);
      if(builder.IsValid()) {
        forest.push_back( Tree<unsigned int>( builder.GetCuts(), builder.GetNEntries(), builder.GetPurities(), builder.GetBoostWeights() ) );
      } else {
        std::cerr << "Terminated boosting at tree " << iTree << " out of " << nTrees << std::endl;
        std::cerr << "Because the last tree was not valid, meaning it couldn't find an optimal cut." << std::endl;
        std::cerr << "This can happen if you do a large number of boosting steps." << std::endl;
        break;
      }
    }

  }

  void ForestBuilder::prepareEventSample(EventSample &sample, double randRatio, bool sPlot) {

    // Draw a random sample if stochastic gradient boost is used
    // Draw random number [0,1) and compare it to the given ratio. If bigger disable this event by flagging it with 0.
    // If smaller set the flag to 1. This is important! If the flags are != 1, the DecisionTree algorithm will fail.
    const unsigned int nEvents = sample.GetNEvents();
    auto &flags = sample.GetFlags();
    if( randRatio < 1.0 and sPlot) {
      // For an sPlot Training it is important to take always signal and background pairs together into the training!
      for(unsigned int iEvent = 0; iEvent < nEvents / 2 + 1; ++iEvent) {
        int use = (static_cast<float>(rand())/static_cast<float>(RAND_MAX) > randRatio ) ? 0 : 1;
        flags.Set(iEvent, use);
        unsigned int jEvent = static_cast<unsigned int>(static_cast<int>(nEvents) - static_cast<int>(iEvent) - 1);
        flags.Set(jEvent, use);
      }
    } else if( randRatio < 1.0) {
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent)
        flags.Set(iEvent, ( static_cast<float>(rand())/static_cast<float>(RAND_MAX) > randRatio ) ? 0 : 1 );
    } else {
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent)
        flags.Set(iEvent, 1);
    }

  }

  void ForestBuilder::updateEventWeights(EventSample &eventSample) {

    const unsigned int nEvents = eventSample.GetNEvents();
    const unsigned int nSignals = eventSample.GetNSignals();

    const auto &flags = eventSample.GetFlags();
    const auto &values = eventSample.GetValues();
    auto &weights = eventSample.GetWeights();

    // Loop over all events and update FCache
    // If the event wasn't disabled, we can use the flag directly to determine the node of this event
    // If not we have to calculate the node to which this event belongs
    if( forest.size() > 0 ) {
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
        if( flags.Get(iEvent) != 0)
          FCache[iEvent] += shrinkage*forest.back().GetBoostWeight( std::abs(flags.Get(iEvent)) - 1);
        else
          FCache[iEvent] += shrinkage*forest.back().GetBoostWeight( forest.back().ValueToNode(&values.Get(iEvent)) );
      }
    }

    for(unsigned int iEvent = 0; iEvent < nSignals; ++iEvent)
      weights.SetBoostWeight(iEvent, 2.0/(1.0+std::exp(2.0*FCache[iEvent])));
    for(unsigned int iEvent = nSignals; iEvent < nEvents; ++iEvent)
      weights.SetBoostWeight(iEvent, 2.0/(1.0+std::exp(-2.0*FCache[iEvent])));

  }
  
  void ForestBuilder::updateEventWeightsWithFlatnessPenalty(EventSample &eventSample) {

    const unsigned int nEvents = eventSample.GetNEvents();
    const unsigned int nSignals = eventSample.GetNSignals();

    const auto &values = eventSample.GetValues();
    auto &weights = eventSample.GetWeights();

    auto nSpectators = values.GetNSpectators();
    
    // Sort events in order of increasing F Value
    for(unsigned int iEvent = 0; iEvent < nSignals; ++iEvent) {
        signal_event_index_sorted_by_F[iEvent] = {FCache[iEvent], iEvent};
    }
    for(unsigned int iEvent = 0; iEvent < nEvents-nSignals; ++iEvent) {
        bckgrd_event_index_sorted_by_F[iEvent] = {-FCache[iEvent+nSignals], iEvent+nSignals};
    }

    {
        auto first = signal_event_index_sorted_by_F.begin();
        auto last = signal_event_index_sorted_by_F.end();
        std::sort(first, last, compareWithIndex<double>);
    }
    
    {
        auto first = bckgrd_event_index_sorted_by_F.begin();
        auto last = bckgrd_event_index_sorted_by_F.end();
        std::sort(first, last, compareWithIndex<double>);
    }

    double global_weight_below_current_F = 0;
    for(unsigned int iIndex = 0; iIndex < signal_event_index_sorted_by_F.size(); ++iIndex) {
        unsigned int iEvent = signal_event_index_sorted_by_F[iIndex].index;
          
        global_weight_below_current_F += weights.GetOriginalWeight(iEvent);
        double F = global_weight_below_current_F / sums[0];
        double fw = 0.0;

        for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
          const uint64_t uniformBin = values.GetSpectator(iEvent, iSpectator);
          weight_below_current_F_per_uniform_bin[iSpectator][uniformBin] += weights.GetOriginalWeight(iEvent);
          double F_bin = weight_below_current_F_per_uniform_bin[iSpectator][uniformBin] / (uniform_bin_weight_signal[iSpectator][uniformBin] * sums[0]);

          fw += (F_bin - F);
        }
        fw *= flatnessLoss;
        weights.SetFlatnessWeight(iEvent, fw);

    }

    for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
      for(uint64_t iUniformBin = 0; iUniformBin < weight_below_current_F_per_uniform_bin[iSpectator].size(); ++iUniformBin) {
        weight_below_current_F_per_uniform_bin[iSpectator][iUniformBin] = 0.0;
      }
    }
    
    global_weight_below_current_F = 0;
    
    for(unsigned int iIndex = 0; iIndex < bckgrd_event_index_sorted_by_F.size(); ++iIndex) {
        unsigned int iEvent = bckgrd_event_index_sorted_by_F[iIndex].index;
          
        global_weight_below_current_F += weights.GetOriginalWeight(iEvent);
        double F = global_weight_below_current_F / sums[1];
        double fw = 0.0;

        for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
          const uint64_t uniformBin = values.GetSpectator(iEvent, iSpectator);
          weight_below_current_F_per_uniform_bin[iSpectator][uniformBin] += weights.GetOriginalWeight(iEvent);
          double F_bin = weight_below_current_F_per_uniform_bin[iSpectator][uniformBin] / (uniform_bin_weight_bckgrd[iSpectator][uniformBin] * sums[1]);

          fw += (F_bin - F);
        }
        fw *= flatnessLoss;
        weights.SetFlatnessWeight(iEvent, fw);

    }

    for(unsigned int iSpectator = 0; iSpectator < nSpectators; ++iSpectator) {
      for(uint64_t iUniformBin = 0; iUniformBin < weight_below_current_F_per_uniform_bin[iSpectator].size(); ++iUniformBin) {
        weight_below_current_F_per_uniform_bin[iSpectator][iUniformBin] = 0.0;
      }
    }

  }

}

