/*
 * Thomas Keck 2014
 */

#pragma once

#define FastBDT_VERSION_MAJOR 1
#define FastBDT_VERSION_MINOR 2

#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

namespace FastBDT {

  /**
   * Compare function which sorts all NaN values to the left
   */
  template<class Value>
  bool compareIncludingNaN (Value i, Value j) {
      if( std::isnan(i) ) {
          return true;
      }
      // If j is NaN the following line will return false,
      // which is fine in our case.
      return i < j;
  }

  /**
   * Since a decision tree operates only on the order of the feature values, a feature binning
   * is performed to optimise the computation without loosing accuracy.
   * The given values of the feature are sorted and binned into 2^nLevels bins.
   * After this transformation the decision tree operates on the bin-index of the values, instead
   * of the value itself. The number of bins determines the number of cuts performed in a traditional decision tree.
   */
  template<class Value>
    class FeatureBinning {

      public:
        /**
         * Creates a new FeatureBinning which maps the values of a feature to bins
         * @param nLevels number of binning levels, in total 2^nLevels bins are used
         * @param first RandomAccessIterator to the begin of the range of values
         * @param last RandomAcessIterator to the end of the range of values
         */
        template<class RandomAccessIterator>
          FeatureBinning(unsigned int nLevels, RandomAccessIterator first, RandomAccessIterator last) : nLevels(nLevels) {

            std::sort(first, last, compareIncludingNaN<Value>);

            // Wind iterator forward until first finite value
            while( std::isnan(*first) ) {
                first++;
            }

            unsigned int size = last - first;
            
            unsigned long int numberOfDistinctValues = 1;
            for(unsigned int iEvent = 1; iEvent < size; ++iEvent) {
              if(first[iEvent] != first[iEvent-1])
                numberOfDistinctValues++;
            }

            // TODO Use numberOfDistinctValues to calculate nLevels automatically
            if(nLevels == 0) {
              // Choose nLevels automatically
              // Same for nTrees
            }

            // Need only Nbins, altough we store upper and lower boundary as well,
            // however GetNBins counts also the NaN bin, so it really is GetNBins() - 1 + 1
            binning.resize(GetNBins(), 0);
            binning.front() = first[0];
            binning.back()  = first[size-1];

            unsigned int bin_index = 0;
            for(unsigned int iLevel = 0; iLevel < nLevels; ++iLevel) {
              const unsigned int nBins = (1 << iLevel);
              for(unsigned int iBin = 0; iBin < nBins; ++iBin) {
                unsigned int range_index = size/(2 << iLevel) + (iBin*size)/(1 << iLevel);
                // TODO Get rid of this limitations, if too few data choose same value multiple times as boundary
                if( range_index == 0)
                  throw std::runtime_error("Number of binning levels to too big for the amount of data you've provided");
                binning[++bin_index] = first[ range_index ];
              }
            }
          }

        unsigned int ValueToBin(const Value &value) const {

          if( std::isnan(value) )
              return 0;

          unsigned int bin = 0;
          unsigned int index = 1;
          for(unsigned int iLevel = 0; iLevel < nLevels; ++iLevel) {
            if( value >= binning[index] ) {
              bin = 2*bin + 1;
              index = 2*index + 1;
            } else {
              bin = 2*bin;
              index = 2*index;
            }
          }
          // +1 because 0 bin is reserved for NaN values
          return bin+1;

        }

        const Value& GetMin() const { return binning.front(); }
        const Value& GetMax() const { return binning.back(); }
        unsigned int GetNLevels() const { return nLevels; }

        /**
         * Return number of bins == 2^nLevels + 1 (NaN bin)
         */
        unsigned int GetNBins() const { return (1 << nLevels)+1; }

        std::vector<Value> GetBinning() const { return binning; }

      private:
        /**
         * The binning boundaries of the feature. First and last element contain minimum and maximum encountered
         * value of the feature. Everything in between marks the boundaries for the bins. In total 2^nLevels bins are used.
         */
        std::vector<Value> binning;
        unsigned int nLevels;

    };

  class EventWeights {

    public:
      EventWeights(unsigned int nEvents) : weights(nEvents, 1), original_weights(nEvents, 0) { }

      inline float Get(unsigned int iEvent) const { return weights[iEvent] * original_weights[iEvent]; }
      void Set(unsigned int iEvent, const float& weight) {  weights[iEvent] = weight; } 
      
      inline const float& GetOriginal(unsigned int iEvent) const { return original_weights[iEvent]; }
      void SetOriginal(unsigned int iEvent, const float& weight) {  original_weights[iEvent] = weight; } 

      /**
       * Returns the sum of all weights. 0: SignalSum, 1: BckgrdSum, 2: SquareSum
       * @param nSignals number of signal events, to determine which weights are signal weights and which are background weights
       */ 
      std::vector<double> GetSums(unsigned int nSignals) const;  

    private:
      std::vector<float> weights;
      std::vector<float> original_weights;
  };

  /**
   * Stores the flags of the events.
   * The flag is used by the decision tree algorithm to store the number of the node, to which
   * this event belings. If the flag is set to <= 0, the event is ignored during the build of the tree
   * flag == 0 means the event is ignored due to stochastic bagging
   * flag < 0 indicates a missing value in this event, where -flag is the last valid node the event belongs to
   */
  class EventFlags {

    public:
      EventFlags(unsigned int nEvents) : flags(nEvents, 1) { }

      inline const int& Get(unsigned int iEvent) const { return flags[iEvent]; } 
      inline void Set(unsigned int iEvent, int flag) { flags[iEvent] = flag; } 

    private:
      std::vector<int> flags;

  };

  class EventValues {

    public:
      EventValues(unsigned int nEvents, unsigned int nFeatures, const std::vector<unsigned int> &nLevels);

      /**
       * Returns a reference to the iFeature feature of the event at position iEvent. The features of one
       * event are garantueed to be stored consecutively on memory. So &GetValue(iEvent) can be used
       * as a pointer to an array filled with the features of the event at position iEvent.
       * @param iEvent position of the event
       * @param iFeature position of feature of the event
       */
      inline const unsigned int& Get(unsigned int iEvent, unsigned int iFeature=0) const { return values[iEvent*nFeatures + iFeature]; }
      void Set(unsigned int iEvent, const std::vector<unsigned int> &features); 

      inline unsigned int GetNFeatures() const { return nFeatures; }

      inline const std::vector<unsigned int>& GetNBins() const { return nBins; }
      inline const std::vector<unsigned int>& GetNBinSums() const { return nBinSums; }

    private:
      /**
       * This vector stores all values. Since the values are garantueed to be stored consecutively in memory,
       * you can use a pointer to the first feature of a given event, as an array holding all features of a given event.
       */
      std::vector<unsigned int> values;
      unsigned int nFeatures; /* <* Amount of features per event*/
      std::vector<unsigned int> nBins; /**< Number of bins for each feature, therefore maximum numerical value of a feature, 0 bin is reserved for NaN values */
      std::vector<unsigned int> nBinSums; /**< Total number of bins up to this feature, including all bins of previous features, excluding first feature  */

  };

  /**
   * The EventSample contains all training events.
   * An event consists of
   *  Bin-indexes of the feature values
   *  Weight
   *  Flag
   *  Class
   * The class of the events (signal or background) is stored implicitly.
   * The first nSignals events in the values, weights and flags arrays are signal events
   * the rest nBackgrounds events are background events.
   * The values array contains nEvents*nFeatures integer values. Where the features of one
   * event are stored consecutively in the memory.
   */
  class EventSample {

    public:
      /** 
       * Creates new EventSample object with enough space for nEvents events with nFeatures features.
       * The memory for all the data is allocated once in this constructor.
       * @param nEvents number of events
       * @param nFeatures number of features per event
       * @param nLevels number of bin levels
       */
      EventSample(unsigned int nEvents, unsigned int nFeatures, const std::vector<unsigned int> &nLevels) : nEvents(nEvents), nSignals(0), nBckgrds(0),
      weights(nEvents), flags(nEvents), values(nEvents,nFeatures,nLevels) { }

      void AddEvent(const std::vector<unsigned int> &features, float weight, bool isSignal);

      /** 
       * Returns whether or not the event is considered as signal. If you loop over all events, it's not necessary to use this function. Just loop
       * over the first nSignals events, which are signal events, and the last nBackgrounds events, which are background events
       * @param iEvent position of event
       * @return True if event is signal, otherwise false
       */
      inline bool IsSignal(unsigned int iEvent) const { return iEvent < nSignals; }

      inline const EventWeights& GetWeights() const { return weights; }
      inline EventWeights& GetWeights() { return weights; }

      inline const EventFlags& GetFlags() const { return flags; }
      inline EventFlags& GetFlags() { return flags; }

      inline const EventValues& GetValues() const { return values; }


      inline unsigned int GetNEvents() const { return nEvents; } 
      inline unsigned int GetNSignals() const { return nSignals; } 
      inline unsigned int GetNBckgrds() const { return nBckgrds; }

    private:
      unsigned int nEvents; 
      unsigned int nSignals; 
      unsigned int nBckgrds; 

      EventWeights weights;
      EventFlags flags;
      EventValues values;

  };


  class CumulativeDistributions {

    public:
      CumulativeDistributions(unsigned int iLayer, const EventSample& sample);

      inline const float& GetSignal(unsigned int iNode, unsigned int iFeature, unsigned int iBin) const { return signalCDFs[iNode*nBinSums.back() + nBinSums[iFeature] + iBin]; }
      inline const float& GetBckgrd(unsigned int iNode, unsigned int iFeature, unsigned int iBin) const { return bckgrdCDFs[iNode*nBinSums.back() + nBinSums[iFeature] + iBin]; }

      unsigned int GetNFeatures() const { return nFeatures; } 
      unsigned int GetNNodes() const { return nNodes; }
      
      inline const std::vector<unsigned int>& GetNBins() const { return nBins; }

    private:
      /**
       * Calculates cumulative distribution functions for every feature and node in the given level
       * @param iLayer layer of the tree
       * @param sample EventSample for which the cumulative distribution is calculated
       * @param firstEvent begin of the range used to calculated the CDFs
       * @param lastEvent  end of the range used to calculate the CDFs
       */
      std::vector<float> CalculateCDFs(const EventSample &sample, const unsigned int firstEvent, const unsigned int lastEvent) const;

    private:
      unsigned int nFeatures;
      std::vector<unsigned int> nBins; /**< Number of bins for each feature, therefore maximum numerical value of a feature, 0 bin is reserved for NaN values */
      std::vector<unsigned int> nBinSums; /**< Total number of bins up to this feature, including all bins of previous features, excluding first feature  */
      unsigned int nNodes;
      std::vector<float> signalCDFs;
      std::vector<float> bckgrdCDFs;
  };

  /**
   * LossFunction -- GiniIndex
   * @param nSignal number of signal events in the node
   * @param nBackgrd number of background events in the node
   */
  double LossFunction(double nSignal, double nBckgrd);


  struct Cut {

    Cut() : feature(0), index(0), gain(0), valid(false) { }            
    unsigned int feature; /**< The feature on which the cut is performed */
    unsigned int index; /**<  The numerically cut value on which the cut is performed */
    double gain; /**< The separationGain of the cut */
    /**
     * Whether the cut is valid. A cut can become invalid if the related node
     * contains only signal or onyl background events. Or if there's no cut
     * available which improves the separation of the node.
     */
    bool valid; 

  };

  class Node {

    public:

      Node(unsigned int iLayer, unsigned int iNode) : signal(0), bckgrd(0), square(0), iNode(iNode), iLayer(iLayer) { }

      /**
       * Calculates for every node in the layer the best Cut with respect to all possible cuts and features using the given CDFs
       */
      Cut CalculateBestCut(const CumulativeDistributions &CDFs) const;

      void AddSignalWeight(float weight, float original_weight);
      void AddBckgrdWeight(float weight, float original_weight);
      void SetWeights(std::vector<double> weights);

      bool IsInLayer(unsigned int iLayer) const { return this->iLayer == iLayer; }
      unsigned int GetLayer() const { return iLayer; }
      unsigned int GetPosition() const { return (iNode + (1 << iLayer)) - 1; }

      double GetPurity() const { return (signal + bckgrd == 0) ? -1 : signal/(signal + bckgrd); }
      double GetBoostWeight() const;

      void Print() const;
    private:
      double signal; /**< The sum of weights of signal events which belong to this node */
      double bckgrd; /**< The sum of weights of background events which belong to this node */
      double square; /**< The squared sum of weights of events which belong to this node */
      unsigned int iNode; /**< Position of the node in the tree */
      unsigned int iLayer; /**< Layer in which the node is inside the tree */
  };



  /**
   * This class trains a DecisionTree.
   *
   * The tree is trained layer by layer, by using the flag of each event
   * to keep track of the node to which the event belongs.
   * The flag always contains the position+1 of the node, starting with 0+1 for the root node.
   *
   * The numeration of the nodes in the tree looks like:  layer
   *                          0                              0
   *                  1               2                      1
   *              3       4       5        6                 2
   *            7   8   9  10  11   12  13  14               3
   *
   * The number of the first node in the n-th level is (1 << n) the
   * number of the last node is (1 << (n+1)) -1.
   */
  class TreeBuilder {

    public:
      TreeBuilder(unsigned int nLayers, EventSample &sample); 
      void Print() const;

      const std::vector<Cut>& GetCuts() const { return cuts; }

      std::vector<float> GetPurities() const { 
        std::vector<float> purities(nodes.size());
        for(unsigned int i = 0; i < nodes.size(); ++i)
          purities[i] = nodes[i].GetPurity();
        return purities; 
      }

      std::vector<float> GetBoostWeights() const { 
        std::vector<float> boostWeights(nodes.size());
        for(unsigned int i = 0; i < nodes.size(); ++i)
          boostWeights[i] = nodes[i].GetBoostWeight();
        return boostWeights; 
      }

    private: 
      void UpdateCuts(const CumulativeDistributions &CDFs, unsigned int iLayer);
      void UpdateFlags(EventSample &sample);
      void UpdateEvents(const EventSample &sample, unsigned int iLayer);

    private:
      unsigned int nLayers; /**< Number of layers in this tree */
      std::vector<Cut> cuts; /**< The best cut for every node in the tree excluding the leave nodes */
      std::vector<Node> nodes; /**< Information about every node in the tree including the leave nodes */

  };

  class Tree {

    public:
      Tree(const std::vector<Cut> &cuts, const std::vector<float> &purities, const std::vector<float> &boostWeights) : cuts(cuts), purities(purities), boostWeights(boostWeights) { }

      /**
       * Returns the node of a given event
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator> unsigned int ValueToNode(const Iterator &values) const;
      unsigned int GetNNodes() const { return boostWeights.size(); }
      const float& GetPurity(unsigned int iNode) const { return purities[iNode]; }
      const float& GetBoostWeight(unsigned int iNode) const { return boostWeights[iNode]; }
      const Cut& GetCut(unsigned int iNode) const { return cuts[iNode]; }
      const std::vector<Cut>& GetCuts() const { return cuts; }
      const std::vector<float>& GetPurities() const { return purities; }
      const std::vector<float>& GetBoostWeights() const { return boostWeights; }

    private:
      std::vector<Cut> cuts;
      std::vector<float> purities;
      std::vector<float> boostWeights;
  };

  template<class Iterator>
    unsigned int Tree::ValueToNode(const Iterator &values) const {

      // Start with a node 1. The node contains the position of the node
      // the event belongs to.
      unsigned int node = 1;

      while( node < cuts.size() ) {

        auto &cut = cuts[node-1];
        if(not cut.valid)
          break;

        // Return current node if NaN
        if( values[ cut.feature ] == 0 )
          break;

        // Perform the cut of the given node and update the node.
        // Either the event is passed to the left child node (which has
        // the position 2*node in the next layer) or to the right
        // (which has the position 2*node + 1 in the next layer)
        // TODO Do index calculation instead of jump here
        if( values[ cut.feature ] < cut.index ) {
          node = (node << 1);
        } else {
          node = (node << 1) + 1;
        }
      }

      // If we're arrived at the bottom of the tree, this event belongs to the node
      // with the position node in the last layer.
      return node - 1;
    }

  /**
   * This class trains a forest of trees with stochastic gradient boosting.
   */
  class ForestBuilder {

    public:
      ForestBuilder(EventSample &eventSample, unsigned int nTrees, double shrinkage, double randRatio, unsigned int nLayersPerTree, bool sPlot=false);
      void print();

      const std::vector<Tree>& GetForest() const { return forest; }
      double GetF0() const { return F0; }
      double GetShrinkage() const { return shrinkage; }

    private:
      void calculateBoostWeights(EventSample &eventSample);
      void updateEventWeights(EventSample &eventSample);
      void prepareEventSample(EventSample &eventSample, double randRatio, bool sPlot);

    private:
      double shrinkage; /**< The config struct for this DecisionForest*/
      double F0; /** The initial F value. Which basically rewights signal and background events based on their initial proportion in the eventSample. */
      std::vector<double> FCache; /**< Caches the F values for the training events, to spare some time.*/
      std::vector<Tree> forest; /**< Contains all the trees trained by the stochastic gradient boost algorithm*/
  };

  class Forest {

    public:
      Forest() = default;
      Forest(const Forest&) = default;
      Forest& operator=(const Forest &) = default;

      Forest(double shrinkage, double F0) : shrinkage(shrinkage), F0(F0) { }

      void AddTree(const Tree &tree) { forest.push_back(tree); }
      const std::vector<Tree>& GetForest() const { return forest; }
      double GetF0() const { return F0; }
      double GetShrinkage() const { return shrinkage; }

      /**
       * Returns the F value calculated from the DecisionForest for a given event.
       * The F value can be used to calculate the signal probability or the weight
       * of the event in the next boosted training.
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator> double GetF(const Iterator &values) const;

      /**
       * Returns the signal probability of a given event
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator> double Analyse(const Iterator &values) const;


      /**
       * Calculates importance ranking of variables, based on the total separation Gain introduced by this variable.
       */
      std::map<unsigned int, double> GetVariableRanking();

    private:
      double shrinkage;
      double F0;
      std::vector<Tree> forest;

  };

  template<class Iterator>
    double Forest::GetF(const Iterator &values) const {

      // Determines the F value by looping over all trees and
      // summing up the weights of the nodes the event belongs to.
      double F = F0;
      for( auto &tree : forest)
        F += shrinkage*tree.GetBoostWeight( tree.ValueToNode(values) );
      return F;

    }

  template<class Iterator>
    double Forest::Analyse(const Iterator &values) const {

      // Calculate signal probability out of the F value
      return 1.0/(1.0+std::exp(-2*GetF(values)));

    }
}
