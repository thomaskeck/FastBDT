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

}
