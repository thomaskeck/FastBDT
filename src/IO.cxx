/**
 * Thomas Keck 2014
 */

#include "IO.h"

namespace FastBDT {
  
  template<>
  std::ostream& operator<<(std::ostream& stream, const std::vector<float> &vector) {
     stream << vector.size();
     for(const auto &value : vector) {
         if (not std::isfinite(value))
            std::cerr << "Writing a nonfinite value, it won't be possible to read the created weightfile!" << std::endl;
         stream << " " << value;
     }
     stream << std::endl;
     return stream;
  }
  
  template<>
  std::ostream& operator<<(std::ostream& stream, const std::vector<double> &vector) {
     stream << vector.size();
     for(const auto &value : vector) {
         if (not std::isfinite(value))
            std::cerr << "Writing a nonfinite value, it won't be possible to read the created weightfile!" << std::endl;
         stream << " " << value;
     }
     stream << std::endl;
     return stream;
  }

  
  std::ostream& operator<<(std::ostream& stream, const Cut &cut) {
     stream << cut.feature << " ";
     stream << cut.index << " ";
     stream << cut.valid << " ";
     stream << cut.gain;
     stream << std::endl;
     return stream;
  }
  
  std::istream& operator>>(std::istream& stream, Cut &cut) {
     stream >> cut.feature;
     stream >> cut.index;
     stream >> cut.valid;
     stream >> cut.gain;
     return stream;
  }
  
  std::ostream& operator<<(std::ostream& stream, const Tree &tree) {
     const auto &cuts = tree.GetCuts();
     stream << cuts.size() << std::endl;
     for( const auto& cut : cuts ) {
        stream << cut << std::endl;
     }
     stream << tree.GetBoostWeights() << std::endl;
     stream << tree.GetPurities() << std::endl;
     return stream;
  }
  
  Tree readTreeFromStream(std::istream& stream) {
   
      unsigned int size;
      stream >> size;
      std::vector<Cut> cuts(size);
      for(unsigned int i = 0; i < size; ++i) {
        stream >> cuts[i];
      }

      std::vector<float> boost_weights;
      stream >> boost_weights;
      
      std::vector<float> purities;
      stream >> purities;
      
      return Tree(cuts, purities, boost_weights);

  }
  
  std::ostream& operator<<(std::ostream& stream, const Forest &forest) {
     
     stream << forest.GetF0() << std::endl;
     stream << forest.GetShrinkage() << std::endl;

     const auto &trees = forest.GetForest();
     stream << trees.size() << std::endl;
     for(const auto& tree : trees) {
         stream << tree << std::endl;
     }

     return stream;
  }
  
  Forest readForestFromStream(std::istream& stream) {
   
      double F0;
      stream >> F0;

      double shrinkage;
      stream >> shrinkage;

      Forest forest(shrinkage, F0);

      unsigned int size;
      stream >> size;

      for(unsigned int i = 0; i < size; ++i) {
        forest.AddTree(readTreeFromStream(stream));
      }

      return forest;

  }
  
  
}
