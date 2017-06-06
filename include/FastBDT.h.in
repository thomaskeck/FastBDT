/*
 * Thomas Keck 2014
 */

#pragma once

#ifndef ADDITIONAL_INCLUDE_GUARD_BECAUSE_ROOT_IS_SO_STUPID
#define ADDITIONAL_INCLUDE_GUARD_BECAUSE_ROOT_IS_SO_STUPID

#define FastBDT_VERSION_MAJOR @FastBDT_VERSION_MAJOR@
#define FastBDT_VERSION_MINOR @FastBDT_VERSION_MINOR@

#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

namespace FastBDT {

  typedef float Weight;

  /**
   * Compare function which sorts all NaN values to the left
   */
  template<class Value>
  bool compareIncludingNaN (Value i, Value j) {
      if( std::isnan(i) ) {
          if(std::isnan(j)) {
            // If both are NAN i is NOT smaller
            return false;
          } else {
            // In all other cases i is smaller
            return true;
          }
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
         * @param values values of this features
         */
          FeatureBinning(unsigned int nLevels, std::vector<Value> &values) : nLevels(nLevels) {
    
            if(nLevels < 2) {
              throw std::runtime_error("Binning level must be at least two!");
            }

            auto first = values.begin();
            auto last = values.end();

            std::sort(first, last, compareIncludingNaN<Value>);

            // Wind iterator forward until first finite value
            while( std::isnan(*first) and first != last ) {
                first++;
            }

            uint64_t size = last - first;

            // If there was no finite data provided (e.g. all values are NaN)
            // We can (and must) choose an arbitrary binning
            // In this case all boundaries are set to 0 and we return
            if(size == 0) {
              binning.resize(GetNBins(), 0);
              return;
            }
            
            // Need only Nbins, altough we store upper and lower boundary as well,
            // however GetNBins counts also the NaN bin, so it really is GetNBins() - 1 + 1
            binning.resize(GetNBins(), first[0]);
            binning[0] = first[0];
            binning[GetNBins()-1] = first[size-1];
            
            uint64_t numberOfDistinctValues = 1;
            std::vector<Value> temp(GetNBins(), first[size-1]);
            temp[0] = first[0];
            temp[1] = first[0];
            for(uint64_t iEvent = 1; iEvent < size; ++iEvent) {
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

            /**
             * We build up our binning in form of a binary tree.
             * Hence we can perform a fast binary tree search later.
             *
             * The first and the last entry in the binning vector contain the minimum and maximum.
             * The remaining entries in binning contain the boundaries of the search tree layer by layer.
             * 
             * Next we calculate the binary tree layer by layer (each layer has 1 << iLevel bin boundaries,
             * so 1, 2, 4, ...). Keep in mind the the feature-data was sorted.
             *
             * First layer:
             * nBins = 1
             * iLevel = 0
             *
             * So the term (size >> 1) is just the middle of our sorted value, hence
             * we use the median as our first boundary and store it in binning[1].
             *
             * Second layer:
             * nBins = 2
             * iLevel = 1
             *
             * (size >> 2) corresponds to the 1. quartile which we store in binning[2]
             * (size >> 2) + (size >> 1) is the 3. quartile which we store in binning[3]
             *
             * We repeat this procedure until we reached the desired number of layers (nLevels) in our tree.
             */
            uint64_t bin_index = 0;
            for(uint64_t iLevel = 0; iLevel < nLevels; ++iLevel) {
              const uint64_t nBins = (1 << iLevel);
              for(uint64_t iBin = 0; iBin < nBins; ++iBin) {
                // (nBins-1)*size overflow possible, therefore we use uint64_t in the whole function!
                binning[++bin_index] = first[ (size >> (iLevel+1)) + ((iBin*size) >> iLevel) ];
              }
            }
          }

        /**
         * Calculate the bin which corresponds to the given value.
         * Our binning is organized in a binary tree, hence we need O(N_bins) operations to do this
         *
         * We start at index = 1 of the binary tree saved in the binning vector.
         * The boundary saved at position 1 is the median of the distribution.
         * 
         * If the value is lower than the median we traverse to the left side of the binary tree,
         * by multiplying the index with 2
         * If the value is higher than the median we traverse to the right side of the binary tree,
         * by multiplying the index with 2 and adding 1
         *
         * The next position in the binary tree we look at is either 2 (the 1. quartile of the distribution)
         * or 3 (the 3. quartile of the distribution)
         *
         * Hence the binary tree (here shown with 3 layers) is saved using the following indices in the binning vector:
         *                             1
         *              2                             3
         *      4               5             6                7
         *
         *  Now we want to get the bins, so we imagine other layer:
         *  8       9     10        11   12       13     14        15
         *
         *  And shift it by (1 << nLevels) - 1 (the highest index in the binary tree! in the example above this is 7)
         *  1       2      3         4    5        6      7         8
         *
         */
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

        /**
         * Calculate the value (here left boundary) which corresponds to a given bin.
         *
         * This one is a little tricky.
         * 0 bin is reserverd for NAN values, so we return NAN
         * 1 is the underflow bin, so we're return the left boundary which is -inf
         *
         * For other values we do the reverse of the ValueToBin function (obviously :-) )
         *
         * So with start with our bins (see example above for a 3-layer binary tree)
         *  1       2      3         4    5        6      7         8
         *
         * Shift them by (1 << nLevels) - 1 so they look like an additional layer of the binary tree
         *  8       9     10        11   12       13     14        15
         *
         * And divide them by 2 as long as this is possible without a remained! And then once more.
         * This projects our "additional layer" onto the indices in the tree which correspond to the nearest left boundary.
         */
        Value BinToValue(unsigned int bin) const {
          if( bin == 0 )
              return NAN;
          
          if( bin == 1 )
              return -std::numeric_limits<Value>::infinity();

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
         * The binning boundaries of the feature. The bin boundaries are organized in binary tree structure.
         * First and last element contain the minimum and maximum encountered value of the feature.
         * Everything in between marks the boundaries for the bins. In total 2^nLevels bins are used.
         *
         * So an example binning for some values uniformely distributed between 0 and 1 looks like this:
         *                                          0.5
         *
         *                         0.25                           0.75
         *
         *                0.125            0.375          0.625          0.875
         *
         * The final binning vector contains the binary tree layer by layer like this:
         * binning = {0,  0.5,   0.25,   0.75,   0.125,  0.375,  0.625,  0.875,  1.0}
         *
         * Hence we require only log(N_bins) operations to find the correct bin of a value instead of O(N_bins)
         */
        std::vector<Value> binning;
        unsigned int nLevels = 0;

    };
  
    /**
     * Compare function which sorts all NaN values to the left
     */
    template<class Value>
    struct ValueWithWeight {
        Value value;
        Weight weight;
    };

    template<class Value>
    bool compareWithWeightsIncludingNaN (ValueWithWeight<Value> i, ValueWithWeight<Value> j) {
        if( std::isnan(i.value) ) {
          if(std::isnan(j.value)) {
            // If both are NAN i is NOT smaller
            return false;
          } else {
            // In all other cases i is smaller
            return true;
          }
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
          WeightedFeatureBinning(unsigned int _nLevels, std::vector<Value> &values, std::vector<Weight> &weights) {

            this->nLevels = _nLevels;
            std::vector<ValueWithWeight<Value>> values_with_weights;
            values_with_weights.resize(values.size());
            Weight total_weight = 0;
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
            while( std::isnan(first->value) and first != last ) {
                first++;
            }

            unsigned int size = last - first;
            
            // If there was no finite data provided (e.g. all values are NaN)
            // We can (and must) choose an arbitrary binning
            // In this case all boundaries are set to 0 and we return
            if(size == 0) {
              this->binning.resize(this->GetNBins(), 0);
              return;
            }
            
            // Need only Nbins, altough we store upper and lower boundary as well,
            // however GetNBins counts also the NaN bin, so it really is GetNBins() - 1 + 1
            this->binning.resize(this->GetNBins(), 0);
            this->binning.front() = first[0].value;
            this->binning.back()  = first[size-1].value;
            Value last_value = first[size-1].value;
            
            Weight weight_per_bin = total_weight / (this->GetNBins() - 1);
            Weight current_weight = 0;
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
            
            auto first = values.begin();
            auto last = values.end();
            // Wind iterator forward until first finite value
            while( std::isnan(*first) and first != last ) {
                first++;
            }

            unsigned int size = last - first;
            
            // If there was no finite data provided (e.g. all values are NaN)
            // We can (and must) choose an arbitrary binning
            // In this case all boundaries are set to 0 and we return
            if(size == 0) {
              this->binning.resize(this->GetNBins(), 0);
              return;
            }

            Value min = *first;
            Value max = *first;
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
    
    /**
     * Compare function which sorts given some values and keeps track of original position
     */
    template<class Value>
    struct ValueWithIndex {
        Value value;
        unsigned int index;
    };

    template<class Value>
    bool compareWithIndex (ValueWithIndex<Value> i, ValueWithIndex<Value> j) {
        if( std::isnan(i.value) ) {
          if(std::isnan(j.value)) {
            // If both are NAN i is NOT smaller
            return false;
          } else {
            // In all other cases i is smaller
            return true;
          }
        }
        // If j is NaN the following line will return false,
        // which is fine in our case.
        return i.value < j.value;
    }
  
    class PurityTransformation {

      public:
        /**
         * Remaps the bins, so that they are ordered with increasing purity
         * @param nLevels number of binning levels, in total 2^nLevels bins are used, must be the same as given to the FeatureBinning!
         * @param values binned values of this features
         * @param weights event weights
         * @param isSignal truth of each event
         */
          PurityTransformation(unsigned int nLevels, const std::vector<unsigned int> &values, const std::vector<Weight> &weights, const std::vector<bool> &isSignal) {

            nBins =  (1 << nLevels)+1;
            mapping.resize(nBins, 0);

            // Count of signal and background events
            std::vector<Weight> counts(2*nBins, 0);

            for(unsigned int iEvent = 0; iEvent < values.size(); ++iEvent) {
                counts[values[iEvent] + static_cast<int>(isSignal[iEvent])*nBins] += weights[iEvent];
            }
            
            std::vector<ValueWithIndex<Weight>> index_with_purities(nBins);
            for(unsigned int iBin = 0; iBin < nBins; ++iBin) {
                index_with_purities[iBin].index = iBin;
                index_with_purities[iBin].value = counts[iBin + nBins] / (counts[iBin] + counts[iBin + nBins]);
            }
            // We ignore the first bin which contains NaN values, which indicate missing not learned!
            std::sort(index_with_purities.begin() + 1, index_with_purities.end(), compareWithIndex<Weight>);

            // NaN Bin always stays at 0 because it has a special meaning (missing not learned)! 
            mapping[0] = 0;
            for(unsigned int iBin = 1; iBin < nBins; ++iBin) {
                mapping[index_with_purities[iBin].index] = iBin;
            }

          }

          PurityTransformation() {
            nBins = 0;
          }
        
          unsigned int BinToPurityBin(unsigned int bin) const {
            return mapping[bin];
          }
        
          std::vector<unsigned int> GetMapping() const { return mapping; }

          void SetMapping(const std::vector<unsigned int> &_mapping) { mapping = _mapping; }

      private:
          unsigned int nBins;
          std::vector<unsigned int> mapping;

    };

  class EventWeights {

    public:
      EventWeights(unsigned int nEvents) : boost_weights(nEvents, 1), flatness_weights(nEvents, 0), original_weights(nEvents, 0) { }

      inline Weight Get(unsigned int iEvent) const { return boost_weights[iEvent] * original_weights[iEvent]; }
      inline const Weight& GetBoostWeight(unsigned int iEvent) const { return boost_weights[iEvent]; }
      inline const Weight& GetFlatnessWeight(unsigned int iEvent) const { return flatness_weights[iEvent]; }
      inline const Weight& GetOriginalWeight(unsigned int iEvent) const { return original_weights[iEvent]; }
      void SetBoostWeight(unsigned int iEvent, const Weight& weight) {  boost_weights[iEvent] = weight; } 
      void SetFlatnessWeight(unsigned int iEvent, const Weight& weight) { flatness_weights[iEvent] = weight; } 
      void SetOriginalWeight(unsigned int iEvent, const Weight& weight) { original_weights[iEvent] = weight; } 

      /**
       * Returns the sum of all weights. 0: SignalSum, 1: BckgrdSum, 2: SquareSum
       * @param nSignals number of signal events, to determine which weights are signal weights and which are background weights
       */ 
      std::vector<Weight> GetSums(unsigned int nSignals) const;  

    private:
      std::vector<Weight> boost_weights;
      std::vector<Weight> flatness_weights;
      std::vector<Weight> original_weights;
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
      EventValues(unsigned int nEvents, unsigned int nFeatures, unsigned int nSpectators, const std::vector<unsigned int> &nLevels);

      /**
       * Returns a reference to the iFeature feature of the event at position iEvent. The features of one
       * event are garantueed to be stored consecutively on memory. So &GetValue(iEvent) can be used
       * as a pointer to an array filled with the features of the event at position iEvent.
       * @param iEvent position of the event
       * @param iFeature position of feature of the event
       */
      inline const unsigned int& Get(unsigned int iEvent, unsigned int iFeature=0) const { return values[iEvent*(nFeatures+nSpectators) + iFeature]; }
      void Set(unsigned int iEvent, const std::vector<unsigned int> &features); 
      inline const unsigned int& GetSpectator(unsigned int iEvent, unsigned int iSpectator=0) const { return values[iEvent*(nFeatures+nSpectators) + nFeatures + iSpectator]; }

      inline unsigned int GetNFeatures() const { return nFeatures; }
      inline unsigned int GetNSpectators() const { return nSpectators; }

      inline const std::vector<unsigned int>& GetNBins() const { return nBins; }
      inline const std::vector<unsigned int>& GetNBinSums() const { return nBinSums; }

    private:
      /**
       * This vector stores all values. Since the values are garantueed to be stored consecutively in memory,
       * you can use a pointer to the first feature of a given event, as an array holding all features of a given event.
       */
      std::vector<unsigned int> values;
      unsigned int nFeatures; /**< Amount of features per event */
      unsigned int nSpectators; /**< Amount of spectators per event */
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
       * @param nSpectators number of spectators per event
       * @param nLevels number of bin levels
       */
      EventSample(unsigned int nEvents, unsigned int nFeatures, unsigned int nSpectators, const std::vector<unsigned int> &nLevels) : nEvents(nEvents), nSignals(0), nBckgrds(0),
      weights(nEvents), flags(nEvents), values(nEvents,nFeatures,nSpectators,nLevels) { }

      void AddEvent(const std::vector<unsigned int> &features, Weight weight, bool isSignal);

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

      inline const Weight& GetSignal(unsigned int iNode, unsigned int iFeature, unsigned int iBin) const { return signalCDFs[iNode*nBinSums[nFeatures] + nBinSums[iFeature] + iBin]; }
      inline const Weight& GetBckgrd(unsigned int iNode, unsigned int iFeature, unsigned int iBin) const { return bckgrdCDFs[iNode*nBinSums[nFeatures] + nBinSums[iFeature] + iBin]; }

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
      std::vector<Weight> CalculateCDFs(const EventSample &sample, const unsigned int firstEvent, const unsigned int lastEvent) const;

    private:
      unsigned int nFeatures;
      std::vector<unsigned int> nBins; /**< Number of bins for each feature, therefore maximum numerical value of a feature, 0 bin is reserved for NaN values */
      std::vector<unsigned int> nBinSums; /**< Total number of bins up to this feature, including all bins of previous features, excluding first feature  */
      unsigned int nNodes;
      std::vector<Weight> signalCDFs;
      std::vector<Weight> bckgrdCDFs;
  };

  /**
   * LossFunction -- GiniIndex
   * @param nSignal number of signal events in the node
   * @param nBackgrd number of background events in the node
   */
  Weight LossFunction(const Weight &nSignal,const Weight &nBckgrd);


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

      void AddSignalWeight(Weight weight, Weight original_weight);
      void AddBckgrdWeight(Weight weight, Weight original_weight);
      void SetWeights(std::vector<Weight> weights);

      bool IsInLayer(unsigned int iLayer) const { return this->iLayer == iLayer; }
      unsigned int GetLayer() const { return iLayer; }
      unsigned int GetPosition() const { return (iNode + (1 << iLayer)) - 1; }

      Weight GetNEntries() const { return signal + bckgrd; }
      Weight GetPurity() const { return (signal + bckgrd == 0) ? -1 : signal/(signal + bckgrd); }
      Weight GetBoostWeight() const;

      void Print() const;
    private:
      Weight signal; /**< The sum of weights of signal events which belong to this node */
      Weight bckgrd; /**< The sum of weights of background events which belong to this node */
      Weight square; /**< The squared sum of weights of events which belong to this node */
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

      std::vector<Weight> GetPurities() const { 
        std::vector<Weight> purities(nodes.size());
        for(unsigned int i = 0; i < nodes.size(); ++i)
          purities[i] = nodes[i].GetPurity();
        return purities; 
      }

      std::vector<Weight> GetBoostWeights() const { 
        std::vector<Weight> boostWeights(nodes.size());
        for(unsigned int i = 0; i < nodes.size(); ++i)
          boostWeights[i] = nodes[i].GetBoostWeight();
        return boostWeights; 
      }
      
			std::vector<Weight> GetNEntries() const { 
        std::vector<Weight> nEntries(nodes.size());
        for(unsigned int i = 0; i < nodes.size(); ++i)
          nEntries[i] = nodes[i].GetNEntries();
        return nEntries; 
      }

      /**
       * Check if the built tree is valid.
       * If we do too much boosting steps on a hard problem, the weights can become nan,
       * and we won't find a optimal cut anymore. A Forest containing a tree without an optimal cut
       * will always return NAN. Therefore this method indicates if we can still use this tree or not
       * */
      bool IsValid() const {
        return cuts[0].valid and std::isfinite(nodes[0].GetBoostWeight());
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

      Tree(const std::vector<Cut<T>> &cuts, const std::vector<Weight> &nEntries, const std::vector<Weight> &purities, const std::vector<Weight> &boostWeights) : cuts(cuts), nEntries(nEntries), purities(purities), boostWeights(boostWeights) { }

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
      
      /**
       * Returns the node-path of a given event - copy & pasted from ValueToNode!
       * @param values the feature values of the event in an arbitrary iterator supporting operator[]
       */
      template<class Iterator> std::vector<unsigned int> ValueToNodePath(const Iterator &values) const {
          
          // TODO Do reserve here, to speed up push_back
          std::vector<unsigned int> node_path;
          unsigned int node = 1;
          while( node <= cuts.size() ) {
            auto &cut = cuts[node-1];
            if(not cut.valid)
              break;
            const T &value = values[cut.feature];
            if(is_nan<T>(value))
              break;
            node_path.push_back(node-1);
            node = (node << 1) + static_cast<unsigned int>(value >= cut.index);
          }

          return node_path;
      }

      unsigned int GetNNodes() const { return boostWeights.size(); }
      const Weight& GetNEntries(unsigned int iNode) const { return nEntries[iNode]; }
      const Weight& GetPurity(unsigned int iNode) const { return purities[iNode]; }
      const Weight& GetBoostWeight(unsigned int iNode) const { return boostWeights[iNode]; }
      const Cut<T>& GetCut(unsigned int iNode) const { return cuts[iNode]; }
      const std::vector<Cut<T>>& GetCuts() const { return cuts; }
      const std::vector<Weight>& GetNEntries() const { return nEntries; }
      const std::vector<Weight>& GetPurities() const { return purities; }
      const std::vector<Weight>& GetBoostWeights() const { return boostWeights; }
      
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
      std::vector<Weight> nEntries;
      std::vector<Weight> purities;
      std::vector<Weight> boostWeights;
  };


  /**
   * This class trains a forest of trees with stochastic gradient boosting.
   */
  class ForestBuilder {

    public:
      ForestBuilder(EventSample &eventSample, unsigned int nTrees, double shrinkage, double randRatio, unsigned int nLayersPerTree, bool sPlot=false, double flatnessLoss=-1.0);
      void print();

      const std::vector<Tree<unsigned int>>& GetForest() const { return forest; }
      double GetF0() const { return F0; }
      double GetShrinkage() const { return shrinkage; }

    private:
      void calculateBoostWeights(EventSample &eventSample);
      void updateEventWeights(EventSample &eventSample);
      void updateEventWeightsWithFlatnessPenalty(EventSample &eventSample);
      void prepareEventSample(EventSample &eventSample, double randRatio, bool sPlot);

    private:
      double shrinkage; /**< The config struct for this DecisionForest*/
      double flatnessLoss; /**< Flatness loss constant, if <=0 no flatness boost ist used */
      double F0; /** The initial F value. Which basically rewights signal and background events based on their initial proportion in the eventSample. */
      std::vector<Weight> sums; /**< Sum of the original weights for signal and background */
      std::vector<double> FCache; /**< Caches the F values for the training events, to spare some time.*/
      std::vector<Tree<unsigned int>> forest; /**< Contains all the trees trained by the stochastic gradient boost algorithm*/
      std::vector<std::vector<double>> uniform_bin_weight_signal; /**< signal weight of each uniform bin */
      std::vector<std::vector<double>> uniform_bin_weight_bckgrd; /**< background weight of each uniform bin */
      std::vector<std::vector<double>> weight_below_current_F_per_uniform_bin; /**< Weight below the current F value in each uniform bin */
      std::vector<ValueWithIndex<double>> signal_event_index_sorted_by_F; /**< The signal event indices sorted by F */
      std::vector<ValueWithIndex<double>> bckgrd_event_index_sorted_by_F; /**< The background event indices sorted by F */
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
      bool GetTransform2Probability() const { return transform2probability; }

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
       * Calculates importance ranking of variables, based on the total separation gain along the path of the event
       */
      template<class Iterator>
      std::map<unsigned int, double> GetIndividualVariableRanking(const Iterator &values) const {
        std::map<unsigned int, double> ranking;
        for(auto &tree : forest) {
          auto node_path = tree.ValueToNodePath(values);
          for(auto & node : node_path) {
            const auto &cut = tree.GetCut(node);
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


      /**
       * Calculates importance ranking of variables, based on the total separation Gain introduced by this variable.
       */
      std::map<unsigned int, double> GetVariableRanking() const {
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
      double shrinkage = 0.0;
      double F0 = 0.0;
      double F0_div_shrink = 0.0;
      std::vector<Tree<T>> forest;
      bool transform2probability = false;

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
      Forest<T> cleaned_forest(forest.GetShrinkage(), forest.GetF0(), forest.GetTransform2Probability());
      for(auto &tree : forest.GetForest()) {
          cleaned_forest.AddTree(removeFeatureBinningTransformationFromTree(tree, featureBinnings));
      }
      return cleaned_forest;
  }

}

#endif
