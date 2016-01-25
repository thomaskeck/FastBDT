/**
 * Thomas Keck 2016
 */

#include <cstdlib>
#include <iostream> 
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"

void TMVAExample( TString filename ) 
{
   TMVA::Tools::Instance();

   TFile *input = TFile::Open(filename);

   TFile* outputFile = TFile::Open( "TMVA.root", "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D" );

   TTree *signal     = static_cast<TTree*>(input->Get("TMVA_tree"))->CopyTree("isSignal == 1");
   TTree *background = static_cast<TTree*>(input->Get("TMVA_tree"))->CopyTree("isSignal == 0");
   factory->AddSignalTree(signal, 1.0);
   factory->AddBackgroundTree(background, 1.0);

   factory->AddVariable( "chiProb",  'F' );
   factory->AddVariable( "piid",  'F' );
   factory->AddVariable( "dz",  'F' );
   factory->AddVariable( "p",  'F' );
   factory->AddVariable( "distance",  'F' );

   factory->PrepareTrainingAndTestTree("", "", "SplitMode=Block:NormMode=None:!V" );

   gPluginMgr->AddHandler("TMVA@@MethodBase", ".*_FastBDT.*", "TMVA::MethodFastBDT", "TMVAFastBDT", "MethodFastBDT(TMVA::DataSetInfo&,TString)");
   gPluginMgr->AddHandler("TMVA@@MethodBase", ".*FastBDT.*", "TMVA::MethodFastBDT", "TMVAFastBDT", "MethodFastBDT(TString&,TString&,TMVA::DataSetInfo&,TString&)");
   factory->BookMethod(TMVA::Types::kPlugins, "FastBDT", "H:V:NTrees=100:Shrinkage=0.1:RandRatio=0.5:NTreeLayers=3:NCutLevel=10");

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();    
   outputFile->Close();

}
