/**
 * Thomas Keck 2017
 */

#include "Classifier.h"
#include <iostream>
#include <fstream>
#include <sstream>

int main() {

  std::fstream stream("unittest.weightfile", std::ios_base::in);
  FastBDT::Classifier classifier2(stream);

  return 0;
}
