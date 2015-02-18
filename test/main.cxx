#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "FBDT.h"

int main(int argc, char *argv[]) {

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

    std::vector<FeatureBinning<double>> featureBinnings;
    for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
        std::vector<double> feature;
        for(auto &event : data) {
            feature.push_back( event[iFeature] );
        }
        featureBinnings.push_back( FeatureBinning<double>(4, feature.begin(), feature.end() ) );
    }

    EventSample eventSample(data.size(), data[0].size()-1 ,4);
    for(auto &event : data) {
        int _class = int(event.back());
        event.pop_back();
        std::vector<unsigned int> bins(4);
        for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
            bins[iFeature] = featureBinnings[iFeature].ValueToBin( event[iFeature] );
        }
        eventSample.AddEvent(bins, 1.0, _class == 1);
    }

   
   /* 
    TreeBuilder dt(2, eventSample);
    dt.Print();

    Tree tree(dt.GetCuts(), dt.GetPurities(), dt.GetBoostWeights());
    */
    
    ForestBuilder dt(eventSample, 1000, 0.1, 0.5, 3);
    Forest tree( dt.GetShrinkage(), dt.GetF0());
    for( auto t : dt.GetForest() )
        tree.AddTree(t);

    std::cout << "Analyse Data" << std::endl;
    int i = 0;
    for(auto &event : data) {
        i++;
        std::vector<unsigned int> bins(4);
        for(unsigned int iFeature = 0; iFeature < 4; ++iFeature) {
            bins[iFeature] = featureBinnings[iFeature].ValueToBin( event[iFeature] );
        }
        //std::cout << std::fixed << std::setprecision(2) << tree.GetPurity( tree.ValueToNode(bins) ) << " ";
        std::cout << std::fixed << std::setprecision(2) << tree.Analyse(bins) << " ";
        if( i % 50 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
    fs.close();

    return 0;

}
