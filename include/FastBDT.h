/*
 * Thomas Keck 2014
 */

#pragma once

#ifndef ADDITIONAL_INCLUDE_GUARD_BECAUSE_ROOT_IS_SO_STUPID
#define ADDITIONAL_INCLUDE_GUARD_BECAUSE_ROOT_IS_SO_STUPID

#define FastBDT_VERSION_MAJOR 2
#define FastBDT_VERSION_MINOR 0

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
   *
   * For an ordered class with distinct values (in contrast to an continueous feature) one should add only
   * the unique values into the feature binning, to ensure that each value gets its own bin.
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
          FeatureBinning(unsigned int nLevels, std::vector<Value> &values) : nLevels(nLevels) {
    
            if(nLevels < 2) {
              throw std::runtime_error("Binning level must be at least two!");
            }

            auto first = values.begin();
            auto last = values.end();

            std::sort(first, last, compareIncludingNaN<Value>);

            // Wind iterator forward until first finite value
            while( std::isnan(*first) ) {
                first++;
            }

            unsigned int size = last - first;
            
            // Need only Nbins, altough we store upper and lower boundary as well,
            // however GetNBins counts also the NaN bin, so it really is GetNBins() - 1 + 1
            binning.resize(GetNBins(), first[0]);
            binning[0] = first[0];
            binning[GetNBins()-1] = first[size-1];

            unsigned long int numberOfDistinctValues = 1;
            std::vector<Value> temp(GetNBins(), first[size-1]);
            temp[0] = first[0];
            temp[1] = first[0];
            for(unsigned int iEvent = 1; iEvent < size; ++iEvent) {
              if(first[iEvent] != first[iEvent-1]) {
                if(numberOfDistinctValues < GetNBins() - 2) {
                  temp[numberOfDistinctValues+1] = first[iEvent];
                }
                numberOfDistinctValues++;
              }
            }
            // Uniquefy the data if there are only a "few" (less than number of bins) unique values
            if(numberOfDistinctValues <= GetNBins() - 2) {
              first = temp.begin();
              last = temp.end();
              size = last - first;
            }

            // TODO Choose nLevels automatically if nLevels == 0

            unsigned int bin_index = 0;
            for(unsigned int iLevel = 0; iLevel < nLevels; ++iLevel) {
              const unsigned int nBins = (1 << iLevel);
              for(unsigned int iBin = 0; iBin < nBins; ++iBin) {
                binning[++bin_index] = first[ (size >> (iLevel+1)) + ((iBin*size) >> iLevel) ];
              }
            }
          }

        unsigned int ValueToBin(const Value &value) const {

          if( std::isnan(value) )
              return 0;

          unsigned int index = 1;
          for(unsigned int iLevel = 0; iLevel < nLevels; ++iLevel) {
              index = 2*index + static_cast<unsigned int>(value >= binning[index]);
          }
          // +1 because 0 bin is reserved for NaN values
          return index - (1 << nLevels) + 1;

        }

        Value BinToValue(unsigned int bin) const {
          if( bin == 0 )
              return NAN;
          
          if( bin == 1 )
              return -std::numeric_limits<double>::infinity();
          
          unsigned int index = bin + (1 << nLevels) - 1;
          for(unsigned int iLevel = 0; iLevel < (nLevels-1) and index % 2 == 0; ++iLevel) {
            index /= 2;
          }
          index /= 2;

          return binning[index];
        }

        const Value& GetMin() const { return binning.front(); }
        const Value& GetMax() const { return binning.back(); }
        unsigned int GetNLevels() const { return nLevels; }

        /**
         * Return number of bins == 2^nLevels + 1 (NaN bin)
         */
        unsigned int GetNBins() const { return (1 << nLevels)+1; }

        std::vector<Value> GetBinning() const { return binning; }

        /*
         * Explicitly activate default/copy constructor and assign operator.
         * This was a request of a user.
         */
        FeatureBinning() = default;
        FeatureBinning(const FeatureBinning&) = default;
        FeatureBinning& operator=(const FeatureBinning &) = default;

      protected:
        /**
         * The binning boundaries of the feature. First and last element contain minimum and maximum encountered
         * value of the feature. Everything in between marks the boundaries for the bins. In total 2^nLevels bins are used.
         */
        std::vector<Value> binning;
        unsigned int nLevels;

    };
  
    /**
     * Compare function which sorts all NaN values to the left
     */
    template<class Value>
    struct ValueWithWeight {
        Value value;
        double weight;
    };

    template<class Value>
    bool compareWithWeightsIncludingNaN (ValueWithWeight<Value> i, ValueWithWeight<Value> j) {
        if( std::isnan(i.value) ) {
            return true;
        }
        // If j is NaN the following line will return false,
        // which is fine in our case.
        return i.value < j.value;
    }

    template<class Value>
    class WeightedFeatureBinning : public FeatureBinning<Value> {

      public:
        /**
         * Creates a new FeatureBinning which maps the values of a feature to bins
         * @param nLevels number of binning levels, in total 2^nLevels bins are used
         * @param values values of this features
         * @param weights of the corresponding events
         */
          WeightedFeatureBinning(unsigned int _nLevels, std::vector<Value> &values, std::vector<double> &weights) {

            this->nLevels = _nLevels;
            std::vector<ValueWithWeight<Value>> values_with_weights;
            values_with_weights.resize(values.size());
            double total_weight = 0;
            unsigned long int numberOfDistinctValues = 1;
            for(unsigned int iEvent = 0; iEvent < values.size(); ++iEvent) {
                values_with_weights[iEvent] = {values[iEvent], weights[iEvent]};
                if (not std::isnan(values[iEvent]))
                    total_weight += weights[iEvent];
                if(iEvent > 0 and values[iEvent] != values[iEvent-1]) {
                  numberOfDistinctValues++;
                }
            }
            
            if(numberOfDistinctValues <= this->GetNBins() - 2) {
              FeatureBinning<Value> temp(this->nLevels, values);
              this->binning = temp.GetBinning();
              return;
            }

            auto first = values_with_weights.begin();
            auto last = values_with_weights.end();
            std::sort(first, last, compareWithWeightsIncludingNaN<Value>);

            // Wind iterator forward until first finite value
            while( std::isnan(first->value) ) {
                first++;
            }

            unsigned int size = last - first;
            
            // Need only Nbins, altough we store upper and lower boundary as well,
            // however GetNBins counts also the NaN bin, so it really is GetNBins() - 1 + 1
            this->binning.resize(this->GetNBins(), 0);
            this->binning.front() = first[0].value;
            this->binning.back()  = first[size-1].value;
            Value last_value = first[size-1].value;
            
            double weight_per_bin = total_weight / (this->GetNBins() - 1);
            double current_weight = 0;
            unsigned int bin = 1;
            while(first != last) {
                current_weight += first->weight;
                // Fill next bin boundary with current value if maximum weight for this bin is reached
                // or if we ran out of bin boundaries
                if(current_weight > weight_per_bin and bin < this->GetNBins()) {
                    current_weight -= weight_per_bin;
                    this->binning[bin] = first->value;
                    bin++;
                }
                first++;
            }
            
            // Fill all remaning bins with highest value
            while(bin < this->GetNBins() -1) {
                this->binning[bin] = last_value;
                bin++;
            }
            
            //Resort binning into correct ordering by binning our bins again!
            FeatureBinning<Value> temp(this->nLevels, this->binning);
            this->binning = temp.GetBinning();


          }
    };
    
    template<class Value>
    class EquidistantFeatureBinning : public FeatureBinning<Value> {

      public:
        /**
         * Creates a new FeatureBinning which maps the values of a feature to bins
         * @param nLevels number of binning levels, in total 2^nLevels bins are used
         * @param values values of this features
         * @param weights of the corresponding events
         */
          EquidistantFeatureBinning(unsigned int _nLevels, std::vector<Value> &values) {

            this->nLevels = _nLevels;
            Value min = values[0];
            Value max = values[0];
            for(unsigned int iEvent = 1; iEvent < values.size(); ++iEvent) {
                if(values[iEvent] > max)
                  max = values[iEvent];
                if(values[iEvent] < min)
                  min = values[iEvent];
            }

            Value step = (max - min) / (this->GetNBins() - 1);
            
            this->binning.resize(this->GetNBins(), 0);
            this->binning.front() = min;
            this->binning.back()  = max;
            for(unsigned int iBin = 1; iBin < this->GetNBins() - 1; ++iBin) {
              this->binning[iBin] = iBin*step + min;
            }

            //Resort binning into correct ordering by binning our bins again!
            FeatureBinning<Value> temp(this->nLevels, this->binning);
            this->binning = temp.GetBinning();

          }
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


  template<typename T>
  struct Cut {

    Cut() : feature(0), index(0), gain(0), valid(false) { }            
    unsigned int feature; /**< The feature on which the cut is performed */
    T index; /**<  The numerically cut value on which the cut is performed */
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
      Cut<unsigned int> CalculateBestCut(const CumulativeDistributions &CDFs) const;

      void AddSignalWeight(float weight, float original_weight);
      void AddBckgrdWeight(float weight, float original_weight);
      void SetWeights(std::vector<double> weights);

      bool IsInLayer(unsigned int iLayer) const { return this->iLayer == iLayer; }
      unsigned int GetLayer() const { return iLayer; }
      unsigned int GetPosition() const { return (iNode + (1 << iLayer)) - 1; }

      double GetNEntries() const { return signal + bckgrd; }
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

      const std::vector<Cut<unsigned int>>& GetCuts() const { return cuts; }

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
      
			std::vector<float> GetNEntries() const { 
        std::vector<float> nEntries(nodes.size());
        for(unsigned int i = 0; i < nodes.size(); ++i)
          nEntries[i] = nodes[i].GetNEntries();
        return nEntries; 
      }

    private: 
      void UpdateCuts(const CumulativeDistributions &CDFs, unsigned int iLayer);
      void UpdateFlags(EventSample &sample);
      void UpdateEvents(const EventSample &sample, unsigned int iLayer);

    private:
      unsigned int nLayers; /**< Number of layers in this tree */
      std::vector<Cut<unsigned int>> cuts; /**< The best cut for every node in the tree excluding the leave nodes */
      std::vector<Node> nodes; /**< Information about every node in the tree including the leave nodes */

  };
      
  /**
   * Check if given value (either float or double) is NaN
   */
  template<class T>
  bool is_nan(const T &value) {
    return std::isnan(value);
  }
  template<>
  bool is_nan(const unsigned int &value);

  template<typename T>
  class Tree {

    public:
      /*
       * Explicitly activate default/copy constructor and assign operator.
       * This was a request of a user.
       */
      Tree() = default;
      Tree(const Tree&) = default;
      Tree& operator=(const Tree &) = default;

      Tree(const std::vector<Cut<T>> &cuts, const std::vector<float> &nEntries, const std::vector<float> &purities, const std::vector<float> &boostWeights) : cuts(cuts), nEntries(nEntries), purities(purities), boostWeights(boostWeights) { }

      /**
       * Returns the node of a given event
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator> unsigned int ValueToNode(const Iterator &values) const {
          // Start with a node 1. The node contains the position of the node
          // the event belongs to.
          unsigned int node = 1;

          while( node <= cuts.size() ) {

            auto &cut = cuts[node-1];
            // Return current node if NaN
            if(not cut.valid)
              break;

            const T &value = values[cut.feature];
            if(is_nan<T>(value))
              break;

            // Perform the cut of the given node and update the node.
            // Either the event is passed to the left child node (which has
            // the position 2*node in the next layer) or to the right
            // (which has the position 2*node + 1 in the next layer)
            node = (node << 1) + static_cast<unsigned int>(value >= cut.index);
          }

          // If we're arrived at the bottom of the tree, this event belongs to the node
          // with the position node in the last layer.
          return node - 1;
      }

      unsigned int GetNNodes() const { return boostWeights.size(); }
      const float& GetNEntries(unsigned int iNode) const { return nEntries[iNode]; }
      const float& GetPurity(unsigned int iNode) const { return purities[iNode]; }
      const float& GetBoostWeight(unsigned int iNode) const { return boostWeights[iNode]; }
      const Cut<T>& GetCut(unsigned int iNode) const { return cuts[iNode]; }
      const std::vector<Cut<T>>& GetCuts() const { return cuts; }
      const std::vector<float>& GetNEntries() const { return nEntries; }
      const std::vector<float>& GetPurities() const { return purities; }
      const std::vector<float>& GetBoostWeights() const { return boostWeights; }
      
      void Print() const {
  
        std::cout << "Start Printing Tree" << std::endl;

        for(auto &cut : cuts) {
          std::cout << "Index: " << cut.index << std::endl;
          std::cout << "Feature: " << cut.feature << std::endl;
          std::cout << "Gain: " << cut.gain << std::endl;
          std::cout << "Valid: " << cut.valid << std::endl;
          std::cout << std::endl;
        }
        
        for(auto &p : purities) {
          std::cout << "Purity: " << p << std::endl;
        }
        for(auto &p : boostWeights) {
          std::cout << "BoostWeights: " << p << std::endl;
        }

        std::cout << "Finished Printing Tree" << std::endl;
      }

    private:
      std::vector<Cut<T>> cuts;
      std::vector<float> nEntries;
      std::vector<float> purities;
      std::vector<float> boostWeights;
  };


  /**
   * This class trains a forest of trees with stochastic gradient boosting.
   */
  class ForestBuilder {

    public:
      ForestBuilder(EventSample &eventSample, unsigned int nTrees, double shrinkage, double randRatio, unsigned int nLayersPerTree, bool sPlot=false);
      void print();

      const std::vector<Tree<unsigned int>>& GetForest() const { return forest; }
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
      std::vector<Tree<unsigned int>> forest; /**< Contains all the trees trained by the stochastic gradient boost algorithm*/
  };

  template<typename T>
  class Forest {

    public:
      /*
       * Explicitly activate default/copy constructor and assign operator.
       * This was a request of a user.
       */
      Forest() = default;
      Forest(const Forest&) = default;
      Forest& operator=(const Forest &) = default;

      Forest(double shrinkage, double F0, bool transform2probability) : shrinkage(shrinkage), F0(F0), transform2probability(transform2probability) { F0_div_shrink = F0 / shrinkage; }

      void AddTree(const Tree<T> &tree) { forest.push_back(tree); }
      const std::vector<Tree<T>>& GetForest() const { return forest; }
      double GetF0() const { return F0; }
      double GetShrinkage() const { return shrinkage; }
      double GetTransform2Probability() const { return transform2probability; }

      /**
       * Returns the F value calculated from the DecisionForest for a given event.
       * The F value can be used to calculate the signal probability or the weight
       * of the event in the next boosted training.
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator>
      double GetF(const Iterator &values) const {

          // Determines the F value by looping over all trees and
          // summing up the weights of the nodes the event belongs to.
          double F = F0_div_shrink;
          for( auto &tree : forest) 
            F += tree.GetBoostWeight( tree.ValueToNode(values) );
          return F*shrinkage;
      }

      /**
       * Returns the signal probability of a given event
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator>
      double Analyse(const Iterator &values) const {

          if(not transform2probability)
              return GetF(values);
          // Calculate signal probability out of the F value
          return 1.0/(1.0+std::exp(-2*GetF(values)));

      }


      /**
       * Calculates importance ranking of variables, based on the total separation Gain introduced by this variable.
       */
      std::map<unsigned int, double> GetVariableRanking() {
        std::map<unsigned int, double> ranking;
        for(auto &tree : forest) {
          for(unsigned int iNode = 0; iNode < tree.GetNNodes()/2; ++iNode) {
            const auto &cut = tree.GetCut(iNode);
            if( cut.valid ) {
              if ( ranking.find( cut.feature ) == ranking.end() )
                ranking[ cut.feature ] = 0;
              ranking[ cut.feature ] += cut.gain;
            }
          }
        }

        double norm = 0;
        for(auto &pair : ranking) {
            norm += pair.second;
        }
        for(auto &pair : ranking) {
            pair.second /= norm;
        }

        return ranking;
      }

    private:
      double shrinkage;
      double F0;
      double F0_div_shrink;
      std::vector<Tree<T>> forest;
      bool transform2probability;

  };
  
  template<typename T>
  Cut<T> removeFeatureBinningTransformationFromCut(const Cut<unsigned int> &cut, const std::vector<FeatureBinning<T>> &featureBinnings) {
      Cut<T> cleaned_cut;
      cleaned_cut.feature = cut.feature;
      cleaned_cut.gain = cut.gain;
      cleaned_cut.valid = cut.valid;
      cleaned_cut.index = featureBinnings[cut.feature].BinToValue(cut.index);
      return cleaned_cut;
  }

  template<typename T>
  Tree<T> removeFeatureBinningTransformationFromTree(const Tree<unsigned int> &tree, const std::vector<FeatureBinning<T>> &featureBinnings) {
      std::vector<Cut<T>> cleaned_cuts;
      cleaned_cuts.reserve(tree.GetCuts().size());
      for(auto &cut : tree.GetCuts()) {
        cleaned_cuts.push_back(removeFeatureBinningTransformationFromCut(cut, featureBinnings));
      }
      return Tree<T>(cleaned_cuts, tree.GetNEntries(), tree.GetPurities(), tree.GetBoostWeights());
  }

  template<typename T>
  Forest<T> removeFeatureBinningTransformationFromForest(const Forest<unsigned int> &forest, const std::vector<FeatureBinning<T>> &featureBinnings) {
      Forest<T> cleaned_forest(forest.GetShrinkage(), forest.GetF0());
      for(auto &tree : forest.GetForest()) {
          cleaned_forest.AddTree(removeFeatureBinningTransformationFromTree(tree, featureBinnings));
      }
      return cleaned_forest;
  }

}

#endif
