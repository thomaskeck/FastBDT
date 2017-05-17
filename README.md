# FastBDT

Stochastic Gradient Boosted Decision Trees, usable standalone, and via Python Interface.

# Paper on ArXiv: http://arxiv.org/abs/1609.06119

FastBDT: A speed-optimized and cache-friendly implementation of stochastic gradient-boosted decision trees for multivariate classification

Stochastic gradient-boosted decision trees are widely employed for multivariate classification and regression tasks. This paper presents a speed-optimized and cache-friendly implementation for multivariate classification called FastBDT. FastBDT is one order of magnitude faster during the fitting-phase and application-phase, in comparison with popular implementations in software frameworks like TMVA, scikit-learn and XGBoost. The concepts used to optimize the execution time and performance studies are discussed in detail in this paper. The key ideas include: An equal-frequency binning on the input data, which allows replacing expensive floating-point with integer operations, while at the same time increasing the quality of the classification; a cache-friendly linear access pattern to the input data, in contrast to usual implementations, which exhibit a random access pattern. FastBDT provides interfaces to C/C++ and Python. It is extensively used in the field of high energy physics by the Belle II experiment. 


# Installation

  * cmake .
  * make
  * make install
  * make package (optional to build rpm, deb packages)
  * python3 setup.py install (optional to install the python package)


# Usage

Before you do anything you want to execute the unittests:
  * ./unittest

But usually it should be more convinient to use FastBDT as a library
and integrate FastBDT directly into your application using
  * the C++ shared/static library (see example/CPPExample.cxx),
  * the C shared library,
  * or the Python3 library python/FastBDT.py (see example/PythonExample.py ).


# Further reading
This work is mostly based on the papers by Jerome H. Friedman
  * https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
  * https://statweb.stanford.edu/~jhf/ftp/stobst.pdf

FastBDT also implements the uGB techniques to boost to flatness:
  * https://arxiv.org/abs/1410.4140

