#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "FBDT.h"
#include "IO.h"
using namespace FastBDT;

int main(int argc, char *argv[]) {

    /*
     * First I read in the data from a txt file.
     */
    if( argc <= 1 ) {
        std::cerr << "Please provide a data file" << std::endl;
        return 1;
    }
    std::fstream fs (argv[1], std::fstream::in | std::fstream::out);
    std::string line;
    std::vector< std::vector<double> > data;
    while(std::getline(fs,line)) {
        std::istringstream sin(line);
        double value = 0;
        std::vector<double> row;
        while(sin >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    /*
     * So now the data is stored in the 2D-Vector data,
     * the FastBDT expects binned data! You can use a FeatureBinning for each feature to bin your data.
     * The binning is an equal statistics binning. So the shape of the input data doesn't matter! (So noramlisation or gaussianisation is not necessary)
     *
     * Basically you just create the FeatureBinning using its constructor:
     *  First argument is the 'binning-level'. In total 2^n bins are used for the feature, in this case 2^4 = 16 (all FeatureBinnings need the SAME binning-level)
     *  Second and third arguments are STL RandomAccess iterators (raw pointers should work also I guess)
     */
    std::vector<FeatureBinning<double>> featureBinnings;
    for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
        std::vector<double> feature;
        for(auto &event : data) {
            feature.push_back( event[iFeature] );
        }
        featureBinnings.push_back( FeatureBinning<double>(4, feature.begin(), feature.end() ) );
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
    EventSample eventSample(data.size(), data[0].size()-1 ,4);
    for(auto &event : data) {
        int _class = int(event.back());
        event.pop_back();
        std::vector<unsigned int> bins(4);
        for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
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
    ForestBuilder dt(eventSample, 1000, 0.1, 0.5, 3);

    /**
     * To apply the trained method again we create a Forest using the trees
     * from the ForestBuilder
     */
    Forest forest( dt.GetShrinkage(), dt.GetF0());
    for( auto t : dt.GetForest() )
        forest.AddTree(t);

    /*
     * Again we use the feature binning to bin the data
     * and afterwards use the Analyse function of the forest to
     * calculate the output
     */
    std::cout << "Analyse Data" << std::endl;
    int i = 0;
    for(auto &event : data) {
        i++;
        std::vector<unsigned int> bins(4);
        for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
            bins[iFeature] = featureBinnings[iFeature].ValueToBin( event[iFeature] );
        }
        //std::cout << std::fixed << std::setprecision(2) << forest.GetPurity( forest.ValueToNode(bins) ) << " ";
        std::cout << std::fixed << std::setprecision(2) << forest.Analyse(bins) << " ";
        if( i % 50 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
    fs.close();

    /**
     * Saving featureBinnings and forest into a file:
     */
    std::fstream file("iris.weight", std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
    file << featureBinnings << std::endl;
    file << forest << std::endl; 
    file.close();

    /**
     * And load both again from the file
     */
    file.open("iris.weight", std::ios_base::in);
    std::vector<FeatureBinning<double>> saved_featureBinnings;
    file >> saved_featureBinnings;
    auto saved_forest = readForestFromStream(file);
    file.close();

    // Apply it again
    std::cout << "Analyse Data with saved classifier" << std::endl;
    i = 0;
    for(auto &event : data) {
        i++;
        std::vector<unsigned int> bins(4);
        for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
            bins[iFeature] = saved_featureBinnings[iFeature].ValueToBin( event[iFeature] );
        }
        //std::cout << std::fixed << std::setprecision(2) << forest.GetPurity( forest.ValueToNode(bins) ) << " ";
        std::cout << std::fixed << std::setprecision(2) << saved_forest.Analyse(bins) << " ";
        if( i % 50 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
    return 0;

}
