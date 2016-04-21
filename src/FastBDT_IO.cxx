/**
 * Thomas Keck 2014
 */

#include "FastBDT_IO.h"

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
  
  /**
   * This function reads a Cut from an std::istream
   * @param stream an std::istream reference
   * @param cut containing read data
   */
  template<>
  std::istream& operator>>(std::istream& stream, Cut<float> &cut) {
     stream >> cut.feature;

     // Unfortunately we have to use stod here to correctly parse NaN and Infinity
     // because usualy istream::operator>> doesn't do this!
     std::string index_string;
     stream >> index_string;
     cut.index = std::stof(index_string);
     //stream >> cut.index;
     stream >> cut.valid;
     stream >> cut.gain;
     return stream;
  }
  
  /**
   * This function reads a Cut from an std::istream
   * @param stream an std::istream reference
   * @param cut containing read data
   */
  template<>
  std::istream& operator>>(std::istream& stream, Cut<double> &cut) {
     stream >> cut.feature;

     // Unfortunately we have to use stod here to correctly parse NaN and Infinity
     // because usualy istream::operator>> doesn't do this!
     std::string index_string;
     stream >> index_string;
     cut.index = std::stod(index_string);
     //stream >> cut.index;
     stream >> cut.valid;
     stream >> cut.gain;
     return stream;
  }

}
