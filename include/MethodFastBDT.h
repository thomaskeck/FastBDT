/*
 * Thomas Keck 2014
 */

#pragma once

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_RVersion
#include "RVersion.h"
#endif

#include "FastBDT.h"

namespace TMVA {

   /**
    * TMVA Method for FastBDT Implementation
    * Thhe FastBDT Method can be loaded at runtime via the Plugin-Mechanism of ROOt
    * using this implementation of a TMVA Method
    */
   class MethodFastBDT : public MethodBase {

   public:
      /**
       * Constructor for training and reading
       * @param jobName unkown
       * @param methodTitle title of the method
       * @param theData data which was added to the TMVA Factory
       * @param theOption options string passed by the user to this method
       * @param theTargetDir target directory in ROOT file which stores the information about the training (removed in ROOT 6.08.00)
       */
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,8,0)
      MethodFastBDT( const TString& jobName,
                 const TString& methodTitle,
                 DataSetInfo& theData,
                 const TString& theOption = "");
#else
      MethodFastBDT( const TString& jobName,
                 const TString& methodTitle,
                 DataSetInfo& theData,
                 const TString& theOption = "",
                 TDirectory* theTargetDir = 0 );
#endif

      /**
       * Constructor for calculating BDT-MVA using previously generated decision trees
       * @param theData data which should be classified
       * @param theWeightFile the xml file from which the method reads the weights
       * @param theTargetDir target directory in ROOT file which stores information about the training (removed in ROOT 6.08.00)
       */
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,8,0)
      MethodFastBDT( DataSetInfo& theData,
                 const TString& theWeightFile);
#else
      MethodFastBDT( DataSetInfo& theData,
                 const TString& theWeightFile,
                 TDirectory* theTargetDir = NULL );
#endif

      /**
       * Destroys method by deleteting the forest and the transformer
       */ 
      virtual ~MethodFastBDT();

      /**
       * Method returns if the user selected analysis is available for this method.
       * This method only supports classification with two classes
       * @param type The type of analysis, e.g. Types::kClassification (supported) or Types::kRegression (not supported)
       * @param numberClasses The number of classes ( only supported value is 2 )
       * @param numberTargets The number of targets ( ignored in this method )
       */
      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );


      /**
       * Trains the FastBDT implementation.
       * First converts the given data with the EventTransform object, after this trains DecisionForest on this transformed data
       */
      void Train();

      /**
       * Initialises the members of the method with default values
       */
      void Init();

      /**
       * Deletes a preceding training, and resets als member to default
       */
      void Reset();


      /**
       * Stores this method in the given XML node
       */
      void AddWeightsXMLTo( void* parent ) const;
      
      /**
       * Reads this method from the given XML node
       */
      void ReadWeightsFromXML(void* parent);
      
      /**
       * Use Version 1
       * - Loads FeatureBinnings and rewrites Trees into new featurebinningless format
       */
      void ReadWeightsFromXML_V1(void* parent);
      
      /**
       * Use Version 2
       * - No FeatureBinnings
       * - additional transform2probability feature
       */
      void ReadWeightsFromXML_V2(void* parent);

      /**
       * FIXME
       * Reads weights from stream
       * This method is just a dummy implementation and prints an error message
       * but it works anyway. This method seems to be deprecated.
       */
      using MethodBase::ReadWeightsFromStream;
      void ReadWeightsFromStream( std::istream& istr );

      /**
       * FIXME
       * Returns the value returned by the decision forest for the current event.
       * This method returns the signal probability.
       * The two arugments aren't used at the moment.
       */
      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0);

      /**
       * Creates an importance ranking of the variables
       */
      const Ranking* CreateRanking();

      /**
       * Declares the possible options, help messages and default values
       */
      void DeclareOptions();

      /**
       * Processed the given options if necessary
       */
      void ProcessOptions();

      /**
       * Sets the number of layers of each tree
       * @param d number of layers
       */
      void SetNTreeLayers(Int_t d){fNTreeLayers = d;}

      /**
       * Sets the amount of cuts for each node in each tree
       * @param d amount of cuts
       */
      void SetNCutLevel(Int_t d){fNCutLevel = d;}

      /**
       * Sets the amount of trees used for the forest
       * @param d amount of trees
       */
      void SetNTrees(Int_t d){fNTrees = d;}

      /**
       * Sets the shrinkage factor for the boost algorithm
       * @param s shrinkage
       */
      void SetShrinkage(Double_t s){fShrinkage = s;}

      /**
       * Sets the proportion of events which are randomly chosen to train each tree
       * @param s ratio of events which are chosen for each tree training. Good value is 0.5
       */
      void SetRandRatio(Double_t s){fRandRatio = s;}

      /**
       * Prints a help message for this method
       */
      void GetHelpMessage() const;

   private:

      /**
       * Number of decision trees requested
       */
      Int_t fNTrees;

      /**
       * Learning rate for gradient boost
       */
      Double_t fShrinkage;  

      /**
       * Ratio for stochastic gradient boost
       */
      Double_t fRandRatio;   
      
      /**
       * sPlot flag, needed for stochastic bagging when we perform an sPlot training
       */
      bool fsPlot;   

      /**
       * Transform to probability
       */
      bool transform2probability;
      
      /**
       * Use weighted feature binning
       */
      bool useWeightedFeatureBinning;
      
      /**
       * Use equidistant feature binning
       */
      bool useEquidistantFeatureBinning;

      /**
       * Binning levels used, determines cuts applied in node splitting
       */
      Int_t fNCutLevel;           

      /**
       * Depth of decision trees
       */
      Int_t fNTreeLayers;          

      /**
       * Pointer to the Forest trained by this method
       */
      FastBDT::Forest<double> *fForest;
      
      /**
       * Class Definition. This is used by ROOT to identify this class
       */
      ClassDef(MethodFastBDT,0)  
   };

} 


