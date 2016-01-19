/**
 * Thomas Keck 2014
 */
#include "MethodFastBDT.h"
#include <vector>
#include <fstream>

#include "Riostream.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TObjString.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMVA/Ranking.h"
#include "TMVA/Results.h"
#include "TMVA/ResultsMulticlass.h"

ClassImp(TMVA::MethodFastBDT)

using namespace FastBDT;

TMVA::MethodFastBDT::MethodFastBDT( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData,
                            const TString& theOption,
                            TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kPlugins, methodTitle, theData, theOption, theTargetDir )
   , fNTrees(0)
   , fShrinkage(0)
   , fRandRatio(0)
   , fsPlot(false)
   , fNCutLevel(0)
   , fNTreeLayers(0)
   , fForest(NULL)
{
}

TMVA::MethodFastBDT::MethodFastBDT( DataSetInfo& theData,
                            const TString& theWeightFile,
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( Types::kPlugins, theData, theWeightFile, theTargetDir )
   , fNTrees(0)
   , fShrinkage(0)
   , fRandRatio(0)
   , fsPlot(false)
   , fNCutLevel(0)
   , fNTreeLayers(0)
   , fForest(NULL)
{
}

TMVA::MethodFastBDT::~MethodFastBDT( void )
{
   if( fForest != NULL )
     delete fForest;
}

Bool_t TMVA::MethodFastBDT::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}

void TMVA::MethodFastBDT::DeclareOptions()
{

   DeclareOptionRef(fNTrees=100, "NTrees", "Number of trees in the forest");
   DeclareOptionRef(fNTreeLayers=3,"NTreeLayers","Max depth of the decision tree allowed");
   DeclareOptionRef(fNCutLevel=8, "NCutLevel", "Number of binning level. Determines number of bins in variable range used in finding optimal cut in node splitting.");
   DeclareOptionRef(fShrinkage=1.0, "Shrinkage", "Learning rate for Gradient Boost algorithm");
   DeclareOptionRef(fRandRatio=1.0, "RandRatio", "Ratio for Stochastic Gradient Boost algorithm");
   DeclareOptionRef(fsPlot=false, "sPlot", "Keep signal and background event pairs together during stochastic bagging, should improve an sPlot training, but frankly said: There was no difference in my tests");
}

void TMVA::MethodFastBDT::ProcessOptions()
{
    // Here is the right place for additional processing of the given options.
    // See other TMVA Methods for examples.
}

void TMVA::MethodFastBDT::Init()
{
   // Common initialisation with defaults for the FastBDT-Method
   fNTrees         = 100;
   fNTreeLayers        = 3;
   fNCutLevel          = 100;
   fShrinkage       = 1.0;
   fRandRatio       = 1.0;
   fsPlot           = false;
}


void TMVA::MethodFastBDT::Reset()
{
   // Reset the method, as if it had just been instantiated (forget all training etc.)
   if( fForest != NULL )   
     delete fForest;
   fForest = NULL;

   featureBinnings.clear();
}


void TMVA::MethodFastBDT::Train()
{

  Data()->SetCurrentType(Types::kTraining);
  UInt_t nEvents = Data()->GetNTrainingEvents();
  UInt_t nFeatures = GetNvar();
  
  // First thing is to read out the training data from Data()
  // into an EventSample object.
  featureBinnings.clear();
  std::vector<unsigned int> nBinningLevels;
  for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
      std::vector<double> feature(nEvents,0);
      for (unsigned int iEvent=0; iEvent<nEvents; iEvent++) {
         feature[iEvent] = GetTrainingEvent(iEvent)->GetValue(iFeature);
      }
      featureBinnings.push_back( FeatureBinning<double>(fNCutLevel, feature.begin(), feature.end() ) );
      nBinningLevels.push_back(fNCutLevel);
  }

  unsigned int nEventsPruned = 0;
  for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
     if(std::abs(GetTrainingEvent(iEvent)->GetWeight()) < 1e-7) {
       std::cerr << "Removed event with extremly small value" << std::endl;
       continue;
     }
     nEventsPruned++;
  }

  EventSample eventSample(nEventsPruned, nFeatures, nBinningLevels);

  for(unsigned int iEvent = 0; iEvent < nEvents; ++iEvent) {
      std::vector<unsigned int> bins(nFeatures);
      auto *event = GetTrainingEvent(iEvent);
      if(std::abs(event->GetWeight()) < 1e-7) {
        continue;
      }
      for(unsigned int iFeature = 0; iFeature < nFeatures; ++iFeature) {
          bins[iFeature] = featureBinnings[iFeature].ValueToBin( event->GetValue(iFeature) );
      }
      eventSample.AddEvent(bins, event->GetWeight(), DataInfo().IsSignal(event));
  }
 
  // Create the forest, this also trains the whole forest immediatly
  ForestBuilder builder(eventSample, fNTrees, fShrinkage, fRandRatio, fNTreeLayers, fsPlot);
  fForest = new Forest( builder.GetShrinkage(), builder.GetF0());
  for( auto tree : builder.GetForest() )
      fForest->AddTree(tree);

}

/**
 * This template saves a vector to a xml file. It creates a new child with the given name
 * and stores the vector under this child.
 * @param parent the parent xml node
 * @param name the name of the child node which contains the vector
 * @param vector the vector which shall be stored
 */
template<class T>
void SaveVectorToXML(void *parent, std::string name, const std::vector<T> &vector) {

   // Create a new child
   void *vecxml = TMVA::gTools().AddChild( parent, name.c_str() );
   // Attach the size of the vector as an attribute to this child
   TMVA::gTools().AddAttr( vecxml, "Size", vector.size() );
   // Store all entries of the vector by attaching additional child-entries
   // The index and the value of every entry of the vector are stored as attributes.
   for (unsigned int i = 0; i < vector.size(); ++i) {
      void *entryxml = TMVA::gTools().AddChild( vecxml, "Entry" );
      TMVA::gTools().AddAttr( entryxml, "Index", i );
      TMVA::gTools().AddAttr( entryxml, "Value", vector[i] );
   }

}

/**
 * This template reads a vector from a xml file. 
 * It reads out the child with the given name and fill the given vector with
 * the entries of this child.
 * @param parent the parent xml node
 * @param name the name of the child node which contains the vector
 * @param vector the vector in which the values read from the child are stored
 */
template<class T>
void ReadVectorFromXML(void *parent, std::string name, std::vector<T> &vector) {

   // Get the child with the name of the vector
   void* vecxml = TMVA::gTools().GetChild(parent, name.c_str());

   // Read out the size of the vector and resize the vector to the correct size
   unsigned int size;
   TMVA::gTools().ReadAttr( vecxml, "Size", size );
   vector.resize(size);

   // Loop over all the entries under the child and read out the index and the value
   // of the entries, and fill the vector with them.
   void* entryxml = TMVA::gTools().GetChild(vecxml, "Entry");
   while (entryxml) {
      unsigned int i;
      T v;
      TMVA::gTools().ReadAttr( entryxml, "Index", i );
      TMVA::gTools().ReadAttr( entryxml, "Value", v);
      vector[i] = v;
      entryxml = TMVA::gTools().GetNextChild(entryxml);
   }

}


void TMVA::MethodFastBDT::AddWeightsXMLTo( void* parent ) const
{
   // write weights to XML
   void* wght = TMVA::gTools().AddChild(parent, "Weights");

   TMVA::gTools().AddAttr( wght, "NTrees", fNTrees );
   TMVA::gTools().AddAttr( wght, "Shrinkage", fShrinkage );
   TMVA::gTools().AddAttr( wght, "NCuts", fNCutLevel );
   TMVA::gTools().AddAttr( wght, "NLevels", fNTreeLayers );
   TMVA::gTools().AddAttr( wght, "RandRatio", fRandRatio );
   TMVA::gTools().AddAttr( wght, "sPlot", fsPlot );
   TMVA::gTools().AddAttr( wght, "F0", fForest->GetF0() );

   auto &forest = fForest->GetForest();
   for(unsigned int i = 0; i < forest.size(); ++i) {
      void* trxml = TMVA::gTools().AddChild(wght, "Tree");
      TMVA::gTools().AddAttr( trxml, "iTree", i );
      
      std::vector<unsigned int> cut_features;
      std::vector<unsigned int> cut_indexes;
      std::vector<bool> cut_valids;
      std::vector<double> cut_gains;
      for( auto& cut : forest[i].GetCuts() ) {
        cut_features.push_back( cut.feature );
        cut_indexes.push_back( cut.index );
        cut_valids.push_back( cut.valid );
        cut_gains.push_back( cut.gain );
      }

      SaveVectorToXML(trxml, "CutFeatures", cut_features);
      SaveVectorToXML(trxml, "CutIndexes", cut_indexes);
      SaveVectorToXML(trxml, "CutValids", cut_valids);
      SaveVectorToXML(trxml, "CutGains", cut_gains);
      SaveVectorToXML(trxml, "BoostWeights", forest[i].GetBoostWeights());
      SaveVectorToXML(trxml, "Purities", forest[i].GetPurities());
      SaveVectorToXML(trxml, "NEntries", forest[i].GetNEntries());
   }

   for(unsigned int i = 0; i < featureBinnings.size(); ++i) {
      void* binxml = TMVA::gTools().AddChild(wght, "Binning");
      TMVA::gTools().AddAttr( binxml, "NLevels", featureBinnings[i].GetNLevels() );
      SaveVectorToXML(binxml, "bins", featureBinnings[i].GetBinning());
   }
  
   /*
   std::map<unsigned int, double> ranking = fForest->GetVariableRanking();
   void *rankingxml = TMVA::gTools().AddChild( parent, "Ranking" );
   for (auto &pair : ranking) {
      void *entryxml = TMVA::gTools().AddChild( rankingxml, "Entry" );
      TMVA::gTools().AddAttr( entryxml, "Variable", GetInputLabel(pair.first) );
      TMVA::gTools().AddAttr( entryxml, "Separation", pair.second );
   }
   */

}

void TMVA::MethodFastBDT::ReadWeightsFromXML(void* parent) {

   Reset();

   TMVA::gTools().ReadAttr( parent, "NTrees", fNTrees );
   TMVA::gTools().ReadAttr( parent, "Shrinkage", fShrinkage );
   TMVA::gTools().ReadAttr( parent, "NCuts", fNCutLevel );
   TMVA::gTools().ReadAttr( parent, "NLevels", fNTreeLayers );
   TMVA::gTools().ReadAttr( parent, "RandRatio", fRandRatio );
   TMVA::gTools().ReadAttr( parent, "sPlot", fsPlot );

   double F0;
   TMVA::gTools().ReadAttr( parent, "F0", F0 );
   fForest = new Forest(fShrinkage, F0);

   void* trxml = TMVA::gTools().GetChild(parent, "Tree");
   while (trxml) {
      // Read the tree number, this value isn't used currently
      unsigned int iTree;
      TMVA::gTools().ReadAttr( trxml, "iTree", iTree );
      // Read the weights
      std::vector<unsigned int> cut_features;
      std::vector<unsigned int> cut_indexes;
      std::vector<bool> cut_valids;
      std::vector<double> cut_gains;
      std::vector<float> boost_weights;
      std::vector<float> purities;
      std::vector<float> nEntries;
      ReadVectorFromXML(trxml, "CutFeatures", cut_features);
      ReadVectorFromXML(trxml, "CutIndexes", cut_indexes);
      ReadVectorFromXML(trxml, "CutValids", cut_valids);
      ReadVectorFromXML(trxml, "CutGains", cut_gains);
      ReadVectorFromXML( trxml, "BoostWeights", boost_weights);
      ReadVectorFromXML( trxml, "Purities", purities);
      ReadVectorFromXML( trxml, "NEntries", nEntries);

      std::vector<Cut> cuts;
      for(unsigned int i = 0; i < cut_features.size(); ++i) {
        Cut cut;
        cut.feature = cut_features[i];
        cut.index = cut_indexes[i];
        cut.valid = cut_valids[i];
        cut.gain = cut_gains[i];
        cuts.push_back(cut);
      }
      fForest->AddTree(Tree(cuts, nEntries, purities, boost_weights));

      trxml = TMVA::gTools().GetNextChild(trxml, "Tree");
   }

   void* binxml = TMVA::gTools().GetChild(parent, "Binning");
   while (binxml) {
      unsigned int nLevels;
      TMVA::gTools().ReadAttr( binxml, "NLevels", nLevels );
      std::vector<double> bins;
      ReadVectorFromXML(binxml, "bins", bins);
      featureBinnings.push_back(FeatureBinning<double>(nLevels, bins.begin(), bins.end()));
      binxml = TMVA::gTools().GetNextChild(binxml, "Binning");
   }

}

void  TMVA::MethodFastBDT::ReadWeightsFromStream( std::istream&)
{
      Log() << kFATAL << "Reading Weights From Stream is currently not supported" << Endl;
}

Double_t TMVA::MethodFastBDT::GetMvaValue( Double_t*, Double_t*){
 
  // First get the current event and store it in a vector 
  std::vector<unsigned int> bins(GetNvar(),0);
  const Event* ev = Data()->GetEvent();
  for (unsigned int iFeature = 0; iFeature < GetNvar(); iFeature++) {
    bins[iFeature] = featureBinnings[iFeature].ValueToBin( ev->GetValue(iFeature) );
  }

  return fForest->Analyse(bins);

}

const TMVA::Ranking* TMVA::MethodFastBDT::CreateRanking()
{
   // Compute ranking of input variables
   fRanking = new TMVA::Ranking( GetName(), "Variable Importance" );
   std::map<unsigned int, double> ranking = fForest->GetVariableRanking();
   for(auto &pair : ranking ) {
        fRanking->AddRank( TMVA::Rank( GetInputLabel(pair.first) , pair.second) );
   }
   return fRanking;
}

void TMVA::MethodFastBDT::GetHelpMessage() const
{
   Log() << "Nice help message" << Endl;
}


