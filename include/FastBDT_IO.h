/**
 * Thomas Keck 2014
 */

#pragma once
#include "FastBDT.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace FastBDT {
  
  /**
   * Converts from string to float safely
   * Should behave similar to boost::lexical_cast<float>
   * but does not signal if it fails!
   * @param input string containing a float
   */
  float convert_to_float_safely(std::string &input);
  
  /**
   * Converts from string to double safely
   * Should behave similar to boost::lexical_cast<double>
   * but does not signal if it fails!
   * @param input string containing a float
   */
  double convert_to_double_safely(std::string &input);

  /**
   * This template saves a vector to an std::ostream
   * @param stream an std::ostream reference
   * @param vector the vector which shall be stored
   */
  template<class T>
  std::ostream& operator<<(std::ostream& stream, const std::vector<T> &vector) {
     stream << vector.size();
     for(const auto &value : vector) {
         stream << " " << value;
     }
     stream << std::endl;
     return stream;
  }
  
  /**
   * Specialize vector output operator, so it checks for nan and infinity in float/double types
   * Note: I know about http://www.gotw.ca/publications/mill17.htm, SFINAE, but nothing worked for me ...
   *       so I sticked with this simple solution instead of complicated template meta programming
   */
  template<>
  std::ostream& operator<<(std::ostream& stream, const std::vector<float> &vector);
  
  template<>
  std::ostream& operator<<(std::ostream& stream, const std::vector<double> &vector);
  
  /**
   * This template reads a vector from an std::istream
   * @param stream an std::istream reference
   * @param vector the vector containing read data
   */
  template<class T>
  std::istream& operator>>(std::istream& stream, std::vector<T> &vector) {
     unsigned int size;
     stream >> size;
     vector.resize(size);
     for(unsigned int i = 0; i < size; ++i) {
         T temp;
         stream >> temp;
         vector[i] = temp;
     }
     return stream;
  }
  
  template<>
  std::istream& operator>>(std::istream& stream, std::vector<float> &vector);
  
  template<>
  std::istream& operator>>(std::istream& stream, std::vector<double> &vector);

  /**
   * This function saves a Cut to an std::ostream
   * @param stream an std::ostream reference
   * @param cut which shall be stored
   */
  template<class T>
  std::ostream& operator<<(std::ostream& stream, const Cut<T> &cut) {
     stream << cut.feature << std::endl;
     stream.precision(std::numeric_limits<T>::max_digits10);
     stream << std::scientific;
     stream << cut.index << std::endl;
     stream.precision(6);
     stream << cut.valid << std::endl;
     stream << cut.gain;
     stream << std::endl;
     return stream;
  }
  
  /**
   * This function reads a Cut from an std::istream
   * @param stream an std::istream reference
   * @param cut containing read data
   */
  template<class T>
  std::istream& operator>>(std::istream& stream, Cut<T> &cut) {
     stream >> cut.feature;
     stream >> cut.index;
     stream >> cut.valid;
     stream >> cut.gain;
     return stream;
  }
  
  template<>
  std::istream& operator>>(std::istream& stream, Cut<float> &cut);
  
  template<>
  std::istream& operator>>(std::istream& stream, Cut<double> &cut);
  
  
  /**
   * This function saves a Tree to an std::ostream
   * @param stream an std::ostream reference
   * @param tree the tree which shall be stored
   */
  template<class T>
  std::ostream& operator<<(std::ostream& stream, const Tree<T> &tree) {
     const auto &cuts = tree.GetCuts();
     stream << cuts.size() << std::endl;
     for( const auto& cut : cuts ) {
        stream << cut << std::endl;
     }
     stream << tree.GetBoostWeights() << std::endl;
     stream << tree.GetPurities() << std::endl;
     stream << tree.GetNEntries() << std::endl;
     return stream;
  }
  
  
  /**
   * This function reads a Tree from an std::istream
   * @param stream an std::istream reference
   * @preturn tree containing read data
   */
  template<class T>
  Tree<T> readTreeFromStream(std::istream& stream) {
      unsigned int size;
      stream >> size;
      std::vector<Cut<T>> cuts(size);
      for(unsigned int i = 0; i < size; ++i) {
        stream >> cuts[i];
      }

      std::vector<Weight> boost_weights;
      stream >> boost_weights;
      
      std::vector<Weight> purities;
      stream >> purities;
      
			std::vector<Weight> nEntries;
      stream >> nEntries;
      
      return Tree<T>(cuts, nEntries, purities, boost_weights);

  }
  
  /**
   * This function saves a Forest to an std::ostream
   * @param stream an std::ostream reference
   * @param forest the forest which shall be stored
   */
  template<class T>
  std::ostream& operator<<(std::ostream& stream, const Forest<T> &forest) {
     stream << forest.GetF0() << std::endl;
     stream << forest.GetShrinkage() << std::endl;
     stream << forest.GetTransform2Probability() << std::endl;

     const auto &trees = forest.GetForest();
     stream << trees.size() << std::endl;
     for(const auto& tree : trees) {
         stream << tree << std::endl;
     }

     return stream;
  }
  
  /**
   * This function reads a Forest from an std::istream
   * @param stream an std::istream reference
   * @preturn forest containing read data
   */
  template<class T>
  Forest<T> readForestFromStream(std::istream& stream) {
      double F0;
      stream >> F0;

      double shrinkage;
      stream >> shrinkage;
     
      bool transform2probability;
      stream >> transform2probability;

      Forest<T> forest(shrinkage, F0, transform2probability);

      unsigned int size;
      stream >> size;

      for(unsigned int i = 0; i < size; ++i) {
        forest.AddTree(readTreeFromStream<T>(stream));
      }

      return forest;
  }
  
  /**
   * This function saves a PurityTransformation to an std::ostream
   * @param stream an std::ostream reference
   * @param purityTransformation the purity transformation which shall be stored
   */
  std::ostream& operator<<(std::ostream& stream, const PurityTransformation &purityTransformation);
  
  /**
   * This function reads a PurityTransformation from an std::istream
   * @param stream an std::istream reference
   * @param purityTransformation the purity transformation which shall be stored
   */
  std::istream& operator>>(std::istream& stream, PurityTransformation &purityTransformation);
  
  
  /**
   * This function saves a FeatureBinning to an std::ostream
   * @param stream an std::ostream reference
   * @param featureBinning the FeatureBinning which shall be stored
   */
  template<class T>
  std::ostream& operator<<(std::ostream& stream, const FeatureBinning<T> &featureBinning) {
     
     stream << featureBinning.GetNLevels() << std::endl;
     stream << featureBinning.GetBinning() << std::endl;

     return stream;
  }
  
  /**
   * This function reads a FeatureBinning from an std::istream
   * @param stream an std::istream reference
   * @preturn FeatureBinning containing read data
   */
  template<class T>
  FeatureBinning<T> readFeatureBinningFromStream(std::istream& stream) {
  
      unsigned int nLevels;
      stream >> nLevels; 

      std::vector<T> bins;
      stream >> bins;

      return FeatureBinning<T>(nLevels, bins);

  }
 
  /**
   * Overload vector input operator, so it can read in FeatureBinnings
   */
  template<class T>
  std::istream& operator>>(std::istream& stream, std::vector<FeatureBinning<T>> &vector) {
     unsigned int size;
     stream >> size;
     for(unsigned int i = 0; i < size; ++i)
         vector.push_back(readFeatureBinningFromStream<T>(stream));
     return stream;
  }
  
  
}
