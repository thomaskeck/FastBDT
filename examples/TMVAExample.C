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

#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"
#include "TVectorD.h"
#include "TRandom.h"

TVectorD MultiGaus(const TVectorD& parMeans, const TMatrixDSym& cov)
{
  TVectorD genPars(0, parMeans.GetNrows()-1);
  TMatrixDSymEigen eigenvariances(cov);
  for(int iPar = 0; iPar < parMeans.GetNrows(); iPar++) {
    double variance = eigenvariances.GetEigenValues()[iPar];
    genPars[iPar] = sqrt(variance)*gRandom->Gaus(0, 1);
  }
  genPars = eigenvariances.GetEigenVectors() * genPars + parMeans;
  return genPars;
}


void TMVAExample() 
{
   TMVA::Tools::Instance();

	 TTree *data = new TTree("data", "data");
   TVectorD random(0, 5);
   data->Branch("Target", &random[0]);
   data->Branch("FeatureA", &random[1]);
   data->Branch("FeatureB", &random[2]);
   data->Branch("FeatureC", &random[3]);
   data->Branch("FeatureD", &random[4]);
   data->Branch("FeatureE", &random[5]);

   TVectorD means(0, 5);
   for(unsigned int i = 0; i < 6; ++i) {
   		means[i] = static_cast<float>(i);
   }

   TMatrixDSym cov(0, 5);
   std::vector<std::vector<float>> t = {{1.0, 0.8, 0.4, 0.2, 0.1, 0.0},
																		  	{0.8, 1.0, 0.0, 0.0, 0.0, 0.0},
																		  	{0.4, 0.0, 1.0, 0.0, 0.0, 0.0},
																	  		{0.2, 0.0, 0.0, 1.0, 0.0, 0.0},
																		  	{0.1, 0.0, 0.0, 0.0, 1.0, 0.0},
																		  	{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
   for(unsigned int i = 0; i < 6; ++i)
       for(unsigned int j = 0; j < 6; ++j)
          cov(i, j) = t[i][j];
	     
   for(unsigned int i = 0; i < 20000; ++i) {
	     random = MultiGaus(means, cov);
       random[0] = (random[0] > 0) ? 1.0 : 0.0;
       data->Fill();
   }

   TFile* outputFile = TFile::Open( "TMVA.root", "RECREATE" );
   TMVA::Factory *factory = new TMVA::Factory( "TMVAClassification", outputFile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D" );

   TTree *signal     = data->CopyTree("Target == 1");
   TTree *background = data->CopyTree("Target == 0");
   factory->AddSignalTree(signal, 1.0);
   factory->AddBackgroundTree(background, 1.0);

   factory->AddVariable( "FeatureA",  'F' );
   factory->AddVariable( "FeatureB",  'F' );
   factory->AddVariable( "FeatureC",  'F' );
   factory->AddVariable( "FeatureD",  'F' );
   factory->AddVariable( "FeatureE",  'F' );

   factory->PrepareTrainingAndTestTree("", "", "SplitMode=Block:NormMode=None:!V" );

   gPluginMgr->AddHandler("TMVA@@MethodBase", ".*_FastBDT.*", "TMVA::MethodFastBDT", "TMVAFastBDT", "MethodFastBDT(TMVA::DataSetInfo&,TString)");
   gPluginMgr->AddHandler("TMVA@@MethodBase", ".*FastBDT.*", "TMVA::MethodFastBDT", "TMVAFastBDT", "MethodFastBDT(TString&,TString&,TMVA::DataSetInfo&,TString&)");
   factory->BookMethod(TMVA::Types::kPlugins, "FastBDT", "H:V:NTrees=200:Shrinkage=0.1:RandRatio=0.5:NTreeLayers=3:NCutLevel=8:standaloneWeightfileName=additionalWeightfile.fbdt");

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();    
   outputFile->Close();

   /*
    * Some convinience code for the CPPExample, which outputs the eigenvectors
    * and eigenvalues as required by this code
   
   TMatrixDSymEigen eigenvariances(cov);
   std::cout << "std::vector<std::vector<float>> eigenvectors = {";
   for(int i = 5; i >= 0; --i) {
     std::cout << "{";
     for(int j = 5; j >= 0; --j)
       std::cout << eigenvariances.GetEigenVectors()(i,j) << ", ";
     std::cout << "}," << std::endl;
   }
   std::cout << "};" << std::endl;
   
   std::cout << "std::vector<float> eigenvalues = {";
   for(int i = 5; i >= 0; --i) {
     std::cout << eigenvariances.GetEigenValues()(i) << ", ";
   }
   std::cout << "};";

   */
}
