/**
 * Thomas Keck 2014
 */

#include "FastBDT_IO.h"

#include <string>
#include <sstream>

namespace FastBDT {

  float convert_to_float_safely(std::string &input) {
     float result = 0;
     try {
        // stof handles infinity and nan correctly but fails
        // for denormalized values
        result = std::stof(input);
     } catch(...) {
        // stringstream fails for nan and infinity but
        // handles denormalized values correctly.
        std::stringstream stream;
        stream << input;
        stream >> result;
     }
     return result;
  }
  
  double convert_to_double_safely(std::string &input) {
     double result = 0;
     try {
        // stof handles infinity and nan correctly but fails
        // for denormalized values
        result = std::stod(input);
     } catch(...) {
        // stringstream fails for nan and infinity but
        // handles denormalized values correctly.
        std::stringstream stream;
        stream << input;
        stream >> result;
     }
     return result;
  }
  
  template<>
  std::ostream& operator<<(std::ostream& stream, const std::vector<float> &vector) {
     stream << vector.size();
     stream.precision(std::numeric_limits<float>::max_digits10);
     stream << std::scientific;
     for(const auto &value : vector) {
         stream << " " << value;
     }
     stream.precision(6);
     stream << std::endl;
     return stream;
  }
  
  template<>
  std::ostream& operator<<(std::ostream& stream, const std::vector<double> &vector) {
     stream << vector.size();
     stream.precision(std::numeric_limits<double>::max_digits10);
     stream << std::scientific;
     for(const auto &value : vector) {
         stream << " " << value;
     }
     stream.precision(6);
     stream << std::endl;
     return stream;
  }
  
  template<>
  std::istream& operator>>(std::istream& stream, std::vector<float> &vector) {
     unsigned int size;
     stream >> size;
     vector.resize(size);
     for(unsigned int i = 0; i < size; ++i) {
         std::string temp;
         stream >> temp;
         vector[i] = convert_to_float_safely(temp);
     }
     return stream;
  }
  
  template<>
  std::istream& operator>>(std::istream& stream, std::vector<double> &vector) {
     unsigned int size;
     stream >> size;
     vector.resize(size);
     for(unsigned int i = 0; i < size; ++i) {
         std::string temp;
         stream >> temp;
         vector[i] = convert_to_double_safely(temp);
     }
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

     // Unfortunately we have to use our own conversion here to correctly parse NaN and Infinity
     // because usualy istream::operator>> doesn't do this!
     std::string index_string;
     stream >> index_string;
     cut.index = convert_to_float_safely(index_string);
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

     // Unfortunately we have to use our own conversion here to correctly parse NaN and Infinity
     // because usualy istream::operator>> doesn't do this!
     std::string index_string;
     stream >> index_string;
     cut.index = convert_to_double_safely(index_string);
     stream >> cut.valid;
     stream >> cut.gain;
     return stream;
  }
  
  std::ostream& operator<<(std::ostream& stream, const PurityTransformation &purityTransformation) {
      stream << purityTransformation.GetMapping() << std::endl;
      return stream;
  }
  
  std::istream& operator>>(std::istream& stream, PurityTransformation &purityTransformation) {

      std::vector<unsigned int> mapping;
      stream >> mapping;
      purityTransformation.SetMapping(mapping);
      return stream;
  }

}
