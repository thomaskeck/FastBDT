# FastBDT

Stochastic Gradient Boosted Decision Trees, usable standalone, as TMVA Plugin and via Python Interface (experimental)


# Installation

cmake .
make
make install

# Usage

./FastBDTMain

or

python3 examples/FastBDTMain.py

for the python3 version you probably have to set the LD_LIBRARY_PATH

or 

integrate FastBDT directly into your application using the C++ shared/static library,
the C shared library

or

use the TMVA plugin of you're using ROOT.

root examples/TMVAExample.C\(\"files/TMVA.root\"\)
