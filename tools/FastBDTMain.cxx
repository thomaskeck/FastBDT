/**
 * Thomas Keck 2014
 */

#include "FastBDT.h"
#include "FastBDT_IO.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>


std::vector<std::vector<double>> readDataFile(std::string datafile) {

    std::fstream fs (datafile, std::fstream::in);
    if(not fs) {
    	throw std::runtime_error("Couldn't open datafile " + datafile);
    }

	std::string line;
	std::vector< std::vector<double> > data;
	unsigned int numberOfFeatures = 0;
  bool skippedHeader = false;
	while(std::getline(fs, line)) {
    if (not skippedHeader) {
      skippedHeader = true;
      continue;
    }
		std::istringstream sin(line);
		double value = 0;
		std::vector<double> row;
		while(sin >> value) {
			row.push_back(value);
		}
		if(numberOfFeatures == 0) {
			numberOfFeatures = row.size();
		} else if (numberOfFeatures != row.size()) {
			throw std::runtime_error("Not all rows have the same number of columns.");
		}
		data.push_back(row);
	}

    if(data.size() < 2) {
    	throw std::runtime_error("Need at least two samples.");
    }

    if(numberOfFeatures == 0) {
    	throw std::runtime_error("Did not find any data in the provided datafile.");
    } else if(numberOfFeatures == 1) {
    	throw std::runtime_error("Did only find one feature in the provided datafile, I need at least two (since last feature is used as target).");
    }
    
    if(data[0].size() < 2) {
    	throw std::runtime_error("Error during read in of the data.");
    }

	return data;
}

void analyse(const FastBDT::Forest<double> &forest, const std::vector<std::vector<double>> &data) {

    /*
     * Again we use the feature binning to bin the data
     * and afterwards use the Analyse function of the forest to
     * calculate the output
     */
    unsigned int correct = 0;
    unsigned int total = data.size();
    unsigned int signal = 0;
    unsigned int signal_correct = 0;
    unsigned int background = 0;
    unsigned int background_correct = 0;
    
    for(auto &event : data) {
        int _class = int(event.back());

        double p = forest.Analyse(event);
        
        if(_class == 1) {
          signal++;
          if(p > 0.5) {
            correct++;
            signal_correct++;
          }
        } else {
          background++;
          if(p < 0.5) {
            correct++;
            background_correct++;
          }
        }
    }

    std::cerr << "Fraction of correctly categorised samples " << std::fixed << std::setprecision(4) << correct/static_cast<double>(total) << std::endl;
    std::cerr << "Signal Efficiency " << std::fixed << std::setprecision(4) << signal_correct/static_cast<double>(signal) << std::endl;
    std::cerr << "Background Efficiency " << std::fixed << std::setprecision(4) << background_correct/static_cast<double>(background) << std::endl;
    std::cerr << "Signal Purity " << std::fixed << std::setprecision(4) << signal_correct/static_cast<double>(signal_correct + background - background_correct) << std::endl;


}

int train(int argc, char *argv[]) {

	if( argc < 4 ) {
		std::cerr << "Usage: " << argv[0] << " train datafile weightfile [nCuts=4] [nTrees=100] [nLevels=3] [shrinkage=0.1] [randRatio=0.5]" << std::endl;
		return 1;
	}

	std::string datafile(argv[2]);
	std::string weightfile(argv[3]);

    unsigned int nCuts = 4;
    if(argc > 4)
    	nCuts = std::stoul(argv[4]);

    unsigned int nTrees = 100;
    if(argc > 5)
    	nTrees = std::stoul(argv[5]);

    unsigned int nLevels = 3;
    if(argc > 6)
    	nLevels = std::stoul(argv[6]);

    double shrinkage = 0.1;
    if(argc > 7)
    	nLevels = std::stod(argv[7]);

    double randRatio = 0.5;
    if(argc > 8)
    	nLevels = std::stod(argv[8]);

    auto data = readDataFile(datafile);
    unsigned int numberOfFeatures = data[0].size() - 1;
    unsigned int numberOfEvents = data.size();

    /*
     * So now the data is stored in the 2D-Vector data,
     * the FastBDT expects binned data! You can use a FeatureBinning for each feature to bin your data.
     * The binning is an equal statistics binning. So the shape of the input data doesn't matter! (So noramlisation or gaussianisation is not necessary)
     *
     * Basically you just create the FeatureBinning using its constructor:
     *  First argument is the 'binning-level'. In total 2^n bins are used for the feature, in this case 2^4 = 16 (all FeatureBinnings need the SAME binning-level)
     *  Second argument is an STL container
     */
    std::vector<FastBDT::FeatureBinning<double>> featureBinnings;
    std::vector<unsigned int> nBinningLevels;
    for(unsigned int iFeature = 0; iFeature < numberOfFeatures; ++iFeature) {
        std::vector<double> feature;
        feature.reserve(numberOfEvents);
        for(auto &event : data) {
            feature.push_back( event[iFeature] );
        }
        featureBinnings.push_back(FastBDT::FeatureBinning<double>(nCuts, feature));
        nBinningLevels.push_back(nCuts);
    }

    /**
     * FastBDT expects the input data in its own format called EventSample,
     * the EventSample automatically sorts the data into signal and background and allows
     * an optimized and fast access during the training.
     *
     * EventSample needs the total size of the training data, the number of features and the binning-level of the FeatureBinnings.
     * In the example below the data is now binned using the FeatureBinning's ValueToBin method
     * and afterwards added (each row seperatly) into the EventSample using its AddEvent method.
     */
    FastBDT::EventSample eventSample(data.size(), numberOfFeatures, nBinningLevels);
    for(auto &event : data) {
        int _class = int(event.back());
        std::vector<unsigned int> bins(numberOfFeatures);
        for(unsigned int iFeature = 0; iFeature < numberOfFeatures; ++iFeature) {
            bins[iFeature] = featureBinnings[iFeature].ValueToBin( event[iFeature] );
        }
        // First argument is the event data as a std::vector,
        // second argument is the event weight
        // third argument is a bool, true indicates signal, false background
        eventSample.AddEvent(bins, 1.0, _class == 1);
    }


    /**
     * Now the aglorithm is trained via the ForestBuilder,
     * the arguments are nTrees, shrinkage, randRatio and treeDepth
     */
    FastBDT::ForestBuilder dt(eventSample, nTrees, shrinkage, randRatio, nLevels);

    /**
     * To apply the trained method again we create a Forest using the trees
     * from the ForestBuilder
     */
    FastBDT::Forest<double> forest( dt.GetShrinkage(), dt.GetF0(), true);
    for( auto t : dt.GetForest() )
        forest.AddTree(FastBDT::removeFeatureBinningTransformationFromTree(t, featureBinnings));

    analyse(forest, data);

    /**
     * Saving featureBinnings and forest into a file:
     */
    std::fstream file(weightfile, std::ios_base::out | std::ios_base::trunc);
    file << forest << std::endl;
    file.close();


    return 0;
}

int apply(int argc, char *argv[]) {

	if( argc < 4 ) {
		std::cerr << "Usage: " << argv[0] << " train datafile weightfile" << std::endl;
		return 1;
	}

	std::string datafile(argv[2]);
	std::string weightfile(argv[3]);

    auto data = readDataFile(datafile);

    std::fstream file(weightfile, std::ios_base::in);
    if(not file) {
    	throw std::runtime_error("Couldn't open weightfile " + weightfile);
    }

    auto forest = FastBDT::readForestFromStream<double>(file);
    file.close();

    analyse(forest, data);

	return 0;
}


int main(int argc, char *argv[]) {

	std::cerr << "FastBDT Version: " << FastBDT_VERSION_MAJOR << "." << FastBDT_VERSION_MINOR << std::endl;

    if( argc <= 1 ) {
        std::cerr << "Usage: " << argv[0] << " [train|apply]" << std::endl;
        return 1;
    }

    if(std::string("train") == argv[1]) {
    	return train(argc, argv);
    }

    if(std::string("apply") == argv[1]) {
    	return apply(argc, argv);
    }

    std::cerr << "Unknown option " << argv[1] << std::endl;
    return 1;

}

