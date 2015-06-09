/**
 * Thomas Keck 2014
 */

#include "FBDT.h"

#include <iostream>
#include <algorithm>
#include <iomanip>


namespace FastBDT {

  std::vector<double> EventWeights::GetSums(unsigned int nSignals) const {

    // Vectorizing FTW!
    std::vector<double> sums(3,0);
    for(unsigned int i = 0; i < nSignals; ++i) {
      sums[0] += weights[i] * original_weights[i];
      sums[2] += weights[i]*weights[i] * original_weights[i];
    }

    for(unsigned int i = nSignals; i < weights.size(); ++i) {
      sums[1] += weights[i] * original_weights[i];
      sums[2] += weights[i]*weights[i] * original_weights[i];
    }
    return sums;

  }

  void EventValues::Set(unsigned int iEvent, const std::vector<unsigned int> &features) {

    // Check if the feature vector has the correct size
    if(features.size() != nFeatures) {
      throw std::runtime_error("Promised number of features are not provided.");
    }

    // Check if the feature values are in the correct range
    for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
      if( features[iFeature] >= nBins )
        throw std::runtime_error("Promised number of bins is violated.");
    }

    // Now add the new values to the values vector.
    for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
      values[iEvent*nFeatures + iFeature] = features[iFeature];
    }

  }

  void EventSample::AddEvent(const std::vector<unsigned int> &features, float weight, bool isSignal) {

    // First check of we have enough space for an additional event. As the number of
    // events is fixed in the constructor (to avoid time consuming reallocations)
    if(nSignals + nBckgrds == nEvents) {
      throw std::runtime_error("Promised maximum number of events exceeded.");
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
    weights.SetOriginal(index, weight);
    values.Set(index, features);

  }

  double LossFunction(double nSignal, double nBckgrd) {
    // Gini-Index x total number of events (needed to calculate information gain efficiently)!
    if( nSignal <= 0 or nBckgrd <= 0 )
      return 0; 
    return (nSignal*nBckgrd)/(nSignal+nBckgrd);
    //return (nSignal*nBckgrd)/((nSignal+nBckgrd)*(nSignal+nBckgrd));
  }

  CumulativeDistributions::CumulativeDistributions(const unsigned int iLayer, const EventSample &sample) {

    const auto &values = sample.GetValues();
    nFeatures = values.GetNFeatures();
    nBins = values.GetNBins();
    nNodes = (1 << iLayer);

    signalCDFs = CalculateCDFs(sample, 0, sample.GetNSignals());
    bckgrdCDFs = CalculateCDFs(sample, sample.GetNSignals(), sample.GetNEvents());

  }

  std::vector<float> CumulativeDistributions::CalculateCDFs(const EventSample &sample, const unsigned int firstEvent, const unsigned int lastEvent) const {

    const auto &values = sample.GetValues();
    const auto &flags = sample.GetFlags();
    const auto &weights = sample.GetWeights();

    std::vector<float> bins( nNodes*nFeatures*nBins );

    // Fill Cut-PDFs for all nodes in this layer and for every feature
    for(unsigned int iEvent = firstEvent; iEvent < lastEvent; ++iEvent) {
      if( flags.Get(iEvent) < static_cast<int>(nNodes) )
        continue;
      const unsigned int index = (flags.Get(iEvent)-nNodes)*nFeatures*nBins;
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature ) {
        const unsigned int subindex = iFeature*nBins + values.Get(iEvent,iFeature);
        bins[index+subindex] += weights.Get(iEvent);
      }
    }

    // Sum up Cut-PDFs to culumative Cut-PDFs
    for(unsigned int iNode = 0; iNode < nNodes; ++iNode) {
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
        for(unsigned int iBin = 1; iBin < nBins; ++iBin) {
          unsigned int index = iNode*nFeatures*nBins + iFeature*nBins + iBin;
          bins[index] += bins[index-1];
        }
      }
    }

    return bins;
  }

  Cut Node::CalculateBestCut(const CumulativeDistributions &CDFs) const {

    Cut cut;

    const unsigned int nFeatures = CDFs.GetNFeatures();
    const unsigned int nBins = CDFs.GetNBins();

    double currentLoss = LossFunction(signal, bckgrd);
    if( currentLoss == 0 )
      return cut;

    // Loop over all features and cuts and sum up signal and background histograms to cumulative histograms
    for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
      for(unsigned int iCut = 1; iCut < nBins; ++iCut) {
        double s = CDFs.GetSignal(iNode, iFeature, iCut-1);
        double b = CDFs.GetBckgrd(iNode, iFeature, iCut-1);
        double currentGain = currentLoss - LossFunction( signal-s, bckgrd-b ) - LossFunction( s, b );

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

  void Node::AddSignalWeight(float weight, float original_weight) {
    signal += weight;
    square += weight*weight / original_weight;
  }


  void Node::AddBckgrdWeight(float weight, float original_weight) {
    bckgrd += weight;
    square += weight*weight / original_weight;
  }

  void Node::SetWeights(std::vector<double> weights) {
    signal = weights[0];
    bckgrd = weights[1];
    square = weights[2];
  }

  double Node::GetBoostWeight() const {

    double denominator = (2*(signal+bckgrd)-square);
    if( denominator == 0 )
      return 0;
    return (signal - bckgrd)/denominator;

  }

  void Node::Print() const {
    std::cout << "Node: " << iNode << std::endl;
    std::cout << "Layer: " << iLayer << std::endl;
    std::cout << "Signal: " << signal << std::endl;
    std::cout << "Bckgrd: " << bckgrd << std::endl;
    std::cout << "Square: " << square << std::endl;
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
    // Secondly, a flag < 0, disables this event, so it isn't used.
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
      if( flag < 0)
        continue;
      auto &cut = cuts[flag-1];
      if( not cut.valid )
        continue;

      if( values.Get(iEvent, cut.feature ) < cut.index ) {
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
        nodes[flag-1].AddSignalWeight( weights.Get(iEvent), weights.GetOriginal(iEvent) );
      }
    }
    for(unsigned int iEvent = sample.GetNSignals(); iEvent < sample.GetNEvents(); ++iEvent) {
      const int flag = flags.Get(iEvent);
      if( flag >= static_cast<int>(nNodes) ) {
        nodes[flag-1].AddBckgrdWeight( weights.Get(iEvent), weights.GetOriginal(iEvent) );
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

  ForestBuilder::ForestBuilder(EventSample &sample, unsigned int nTrees, double shrinkage, double randRatio, unsigned int nLayersPerTree, bool sPlot) : shrinkage(shrinkage) {

    auto &weights = sample.GetWeights();
    auto sums = weights.GetSums(sample.GetNSignals()); 
    // Calculating the initial F value from the proportion of the number of signal and background events in the sample
    //double average = (static_cast<int>(sample.GetNSignals()) - static_cast<int>(sample.GetNBckgrds()))/static_cast<double>(sample.GetNSignals() + sample.GetNBckgrds());
    double average = (sums[0] - sums[1])/(sums[0] + sums[1]);
    F0 = 0.5*std::log((1+average)/(1-average));

    // Resize the FCache to the number of events, and initalise it with the inital F value
    FCache.resize(sample.GetNEvents(), F0);

    // Reserve enough space for the boost_weights and trees, to avoid reallocations
    forest.reserve(nTrees);

    // Now train config.nTrees!
    for(unsigned int iTree = 0; iTree < nTrees; ++iTree) {
    
      // Update the event weights according to their F value
      updateEventWeights(sample);

      // Prepare the flags of the events
      prepareEventSample( sample, randRatio, sPlot );   

      // Create and train a new train on the sample
      TreeBuilder builder(nLayersPerTree, sample);
      forest.push_back( Tree( builder.GetCuts(), builder.GetPurities(), builder.GetBoostWeights() ) );
    }

  }

  void ForestBuilder::prepareEventSample(EventSample &sample, double randRatio, bool sPlot) {

    // Draw a random sample if stochastic gradient boost is used
    // Draw random number [0,1) and compare it to the given ratio. If bigger disable this event by flagging it with -1.
    // If smaller set the flag to 1. This is important! If the flags are != 1, the DecisionTree algorithm will fail.
    const unsigned int nEvents = sample.GetNEvents();
    auto &flags = sample.GetFlags();
    if( randRatio < 1.0 and sPlot) {
      // For an sPlot Training it is important to take always signal and background pairs together into the training!
      for(unsigned int iEvent = 0; iEvent < nEvents / 2 + 1; ++iEvent) {
        int use = (static_cast<float>(rand())/static_cast<float>(RAND_MAX) > randRatio ) ? -1 : 1;
        flags.Set(iEvent, use);
        unsigned int jEvent = static_cast<unsigned int>(static_cast<int>(nEvents) - static_cast<int>(iEvent) - 1);
        flags.Set(jEvent, use);
      }
    } else if( randRatio < 1.0) {
      for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent)
        flags.Set(iEvent, ( static_cast<float>(rand())/static_cast<float>(RAND_MAX) > randRatio ) ? -1 : 1 );
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
        if( flags.Get(iEvent) >= 0)
          FCache[iEvent] += shrinkage*forest.back().GetBoostWeight( flags.Get(iEvent) - 1);
        else
          FCache[iEvent] += shrinkage*forest.back().GetBoostWeight( forest.back().ValueToNode(&values.Get(iEvent)) );
      }
    }

    for(unsigned int iEvent = 0; iEvent < nSignals; ++iEvent)
      weights.Set(iEvent, 2.0/(1.0+std::exp(2.0*FCache[iEvent])));
    for(unsigned int iEvent = nSignals; iEvent < nEvents; ++iEvent)
      weights.Set(iEvent, 2.0/(1.0+std::exp(-2.0*FCache[iEvent])));

  }

  std::map<unsigned int, double> Forest::GetVariableRanking() {

    std::map<unsigned int, double> ranking;

    for(auto &tree : forest) {
      for(unsigned int iNode = 0; iNode < tree.GetNNodes()/2; ++iNode) {
        const auto &cut = tree.GetCut(iNode);
        if( cut.valid ) {
          if ( ranking.find( cut.feature ) != ranking.end() )
            ranking[ cut.feature ] = 0;
          ranking[ cut.feature ] += cut.gain*std::abs(tree.GetBoostWeight(iNode));
        }
      }
    }

    return ranking;
  }
}
