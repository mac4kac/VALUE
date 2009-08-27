#include "nnfw.h"
#include "nnfwfactory.h"
#include "biasedcluster.h"
#include "simplecluster.h"
#include "fakecluster.h"
#include "dotlinker.h"
#include "copylinker.h"
#include "normlinker.h"
#include "liboutputfunctions.h"
#include "backpropagationalgo.h"
#include "random.h"
#include "time.h"
#include "types.h"
#include <math.h>
#include "ionnfw.h"
#include <vector>
#include <string.h>
#include "propertized.h"
#include "libradialfunctions.h"
#include <stdio.h>



// 23 epochs for learning 2993 inputs = approx 70,000 iterations
#define numInput 2993 // defined in data/srn.data
#define numEpochs 23
#define numTestInput 32 // defined in data/predtest.data
#define learnRate 0.075 

using namespace nnfw;
using namespace std;

//-------- Neural Network Structures
BiasedCluster *in, *hid, *outCat, *out;
SimpleCluster* cont;
DotLinker *l1, *l2, *l3, *l4;
CopyLinker* cplinker1;
BaseNeuralNet* net;
BackPropagationAlgo *bp, *bpCat; 

// IT SOM - Object Identity
BaseNeuralNet* IT_net;
SimpleCluster* IT_input;
NormLinker* IT_l1;
// MT SOM - Experimenter Instruction
BaseNeuralNet* MT_net;
SimpleCluster* MT_input;
NormLinker* MT_l1;
// PFC SOM - Current Goal
BaseNeuralNet* PFC_net;
SimpleCluster* PFC_input;
NormLinker* PFC_l1;

int inputSize=200; // 200 outputs from vision + PFC SOM
int PFC_inputSize=200; // 200 outputs from IT+MT SOM
int IT_inputSize=100; // 2D map of visual input
int MT_inputSize=2; // tasks

double val[16]; // array for 16 inputs read from file
int visionInput[100]; // 2D map of the visual input
double targetOutput[10][16]; // 10 sequences for 16 joints in hand

int bigBallVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int bigBoxVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int smallBallVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int smallBoxVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

int expInstr[2];

int ExpInstr_Graps[2]= {0,1};
int ExpInstr_Cat[2]= {1,0};

int naturalObject[2]= {0,1};
int artefactObject[2]= {1,0};


void testJordan(int i);


class CompetitiveCluster : public SimpleCluster {
	public:
		CompetitiveCluster(int r, int c, const char* name) : SimpleCluster(r*c,name), rows(r), cols(c) {
			outmatrix = new RealMat( outputs(), 0, rows*cols, rows, cols );
			addProperty( "rows", Variant::t_int, this, &CompetitiveCluster::getRows );
			addProperty( "cols", Variant::t_int, this, &CompetitiveCluster::getColums );
			setTypename("CompetitiveCluster");
		}

		CompetitiveCluster( PropertySettings& prop ) : SimpleCluster(prop) {
			Variant& v = prop["rows"];
			if ( v.isNull() ) {
			//nFatal() << "Skata ";
			}
			rows = convertStringTo( v, Variant::t_int ).getInt();
			v = prop["cols"];
			if ( v.isNull() ) {
			//nFatal() << "Skata ";
			}
			cols = convertStringTo( v, Variant::t_int ).getInt();
			outmatrix = new RealMat( outputs(), 0, rows*cols, rows, cols );
			addProperty( "rows", Variant::t_int, this, &CompetitiveCluster::getRows );
			addProperty( "cols", Variant::t_int, this, &CompetitiveCluster::getColums );
			setTypename("CompetitiveCluster");
		};
	
		Variant getRows() {
			return Variant( rows );
		};

		Variant getColums(){
			return Variant ( cols);
		};

		void update() {
			SimpleCluster::update();
			RealMat& refmat = *outmatrix;
			//competition among outputs
			centreX = 0.0;
			centreY = 0.0;
			int cx = 0;
			int cy = 0;
			for( int r=0; r<rows; r++ ) { 
				for( int c=0; c<cols; c++ ) {
					if ( refmat[r][c] > refmat[cx][cy] ) {
						cx = r;
						cy = c;
					}
					//centreX += c*refmat[r][c]; //find the center position
					//centreY += r*refmat[r][c];
				}
			}
			//centreX /= numNeurons();
			//centreY /= numNeurons();
			centreX = (Real)cx;
			centreY = (Real)cy;
		};
	
		void getCentre( Real& x, Real& y ) {
			x = centreX;
			y = centreY;
		};	

	private:
		RealMat* outmatrix;
		int rows;
		int cols;
		Real centreX;
		Real centreY;
};

Creator <CompetitiveCluster> c;
bool dummy = Factory::registerCluster(c,"CompetitiveCluster");

CompetitiveCluster* PFC_m1;
CompetitiveCluster* IT_m1;
CompetitiveCluster* MT_m1;


int main( int , char*[] ) {
 	
 	Random::setSeed(time(0));
    net = new BaseNeuralNet();
    
    // --- LOAD THE NEURAL NET VIA XML 
	net = loadXML( "net.xml" );
	// --- SET THE DIFFERENT LAYERS ACCORDINGLY
	in = (BiasedCluster*)net->getByName("Input");
	cont = (SimpleCluster*)net->getByName("Context");
	hid = (BiasedCluster*)net->getByName("Hidden");
	outCat = (BiasedCluster*)net->getByName("OutputCat");
	out = (BiasedCluster*)net->getByName("Output");
	
	l1 = (DotLinker*)net->getByName("In2Hid");
	l2 = (DotLinker*)net->getByName("Cont2Hid");
	l3 = (DotLinker*)net->getByName("Hid2OutCat");
	l4 = (DotLinker*)net->getByName("Hid2Out");
	cplinker1 = (CopyLinker*)net->getByName("Out2Cont");	
	
	// --- RANDOMIZE THE NET
	net->randomize( -1.0, 1.0 );
	// --- REVERSE THE ORDER OF THE NET
	UpdatableVec rev_ord(net->order().size()); 
	rev_ord.assign_reverse(net->order());
	// --- CREATE THE BACKPROPAGATION ALGORITHM
	bp = new BackPropagationAlgo (net, rev_ord, learnRate);
	bpCat = new BackPropagationAlgo (net, rev_ord, learnRate);

	// --- LOAD the IT SOM ----------------------------------------------------------------- //
	IT_net = loadXML("data/IT_SOM.xml");

	const ClusterVec& cl = IT_net->clusters();
	for( nnfw::u_int i=0; i<cl.size(); i++ ){                
    	 cl[i]->inputs().zeroing();
    	 cl[i]->outputs().zeroing();                                
 	}
	// --- SET THE DIFFERENT LAYERS ACCORDINGLY
	IT_input = (SimpleCluster*)IT_net->getByName("IT_input");
	IT_m1 = (CompetitiveCluster*)IT_net->getByName("IT_map");
	IT_l1 = (NormLinker*)IT_net->getByName("IT_link");
	// ------------------------------------------------------------------------------------- //

	// --- LOAD the MT SOM ----------------------------------------------------------------- //
	MT_net = loadXML("data/MT_SOM.xml");

	const ClusterVec& cl1 = MT_net->clusters();
	for( nnfw::u_int i=0; i<cl1.size(); i++ ){                
    	 cl1[i]->inputs().zeroing();
    	 cl1[i]->outputs().zeroing();                                
 	}
	
	// --- SET THE DIFFERENT LAYERS ACCORDINGLY
	MT_input = (SimpleCluster*)MT_net->getByName("MT_input");
	MT_m1 = (CompetitiveCluster*)MT_net->getByName("MT_map");
	MT_l1 = (NormLinker*)MT_net->getByName("MT_link");
	// ------------------------------------------------------------------------------------- //

	// --- LOAD the PFC SOM ---------------------------------------------------------------- //
	PFC_net = loadXML("data/PFC_SOM.xml");

	const ClusterVec& cl2 = PFC_net->clusters();
	for( nnfw::u_int i=0; i<cl2.size(); i++ ){                
    	 cl2[i]->inputs().zeroing();
    	 cl2[i]->outputs().zeroing();                                
 	}
	
	// --- SET THE DIFFERENT LAYERS ACCORDINGLY
	PFC_input = (SimpleCluster*)PFC_net->getByName("PFC_input");
	PFC_m1 = (CompetitiveCluster*)PFC_net->getByName("PFC_map");
	PFC_l1 = (NormLinker*)PFC_net->getByName("PFC_link");
	// ------------------------------------------------------------------------------------- //

	// --- PRINT PRECISION
	cout.precision( 5 );

	RealVec Data = in->numNeurons();
	RealVec PFC_Data = PFC_input->numNeurons();
	RealVec IT_Data = IT_input->numNeurons();
	RealVec MT_Data = MT_input->numNeurons();

	FILE *DATA;
	FILE *f = fopen("results/trainError.data","w"); // fopen("data/SBox/graspingNormSeq.data","w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}
	FILE *f1 = fopen("results/categoriseError.data","w"); 
	if (!f1) {printf("can't open config file for writing\n");exit(1);}

	// --- SETTING UP TRAINING INPUT ------------------------------------------------------------------------------- //
	int task;	
	int objectViewed;
	int graspSeq; // = -1;
	int trainingItem; // 3 modes of training: 0 = all PFC outputs set to 0, 1 = grasping task, 2 = categorising task

	int train0 = 3000; //3333; // Number of training instances for each training mode
	int train1 = 3000; //5000;
	//int train2 = 10000;
	
	int numIteration = 12000; //30000; //60000;
	int firstTrainingPhase = 6000; //3334; //10000; //numIteration/3;
	int secondTrainingPhase = firstTrainingPhase + train0 + train1;

	int trainingInput[numIteration];

	for (int t=0; t < firstTrainingPhase; t++) 
		trainingInput[t] = 1;

	for (int t = 0; t < train0; t++)
		trainingInput[firstTrainingPhase+t] = 1;
	for (int t = 0; t < train1; t++)
		trainingInput[firstTrainingPhase+train0+t] = 2;

	//for (int t = 0; t < train0; t++)
	//	trainingInput[secondTrainingPhase+t] = 0;
	//for (int t = 0; t < train1; t++)
	//	trainingInput[secondTrainingPhase+train0+t] = 1;
	//for (int t = 0; t < train2; t++)
	//	trainingInput[secondTrainingPhase+train0+train1+t] = 2;

	//for(int i=0;i<numIteration;++i)
    //  cout<<trainingInput[i];
	//cout << endl;

	random_shuffle(trainingInput + firstTrainingPhase, trainingInput + secondTrainingPhase);

	//for(int i=0;i<numIteration;++i)
    //    cout<<trainingInput[i];
	//cout << endl;

	//random_shuffle(trainingInput + secondTrainingPhase, trainingInput + numIteration);

	//for(int i=0;i<numIteration;++i) {
    //   cout<<trainingInput[i];
	//}
	//cout << endl;

	int bbal=0; int bbox=0; int sbal=0; int sbox=0;
	// ------------------------------------------------------------------------------------------------------------- //

	for (int num = 0; num < numIteration; num++) {
		objectViewed = Random::flatInt(0,4);
		//objectViewed = 0;
		graspSeq = Random::flatInt(0,4);
		//trainingItem = trainingInput[num];
		trainingItem = 1;
/*		
		if (num < firstTrainingPhase) {
			task  = 0;
			trainingItem = 0;
		}
		if (num > firstTrainingPhase && num < secondTrainingPhase) {
			task  = 0;
			trainingItem = Random::flatInt(0,2);
		}
		if (num > secondTrainingPhase) {
			trainingItem = Random::flatInt(0,3);
			if (trainingItem == 2) {task = 1;}
			else {task = 0;}
		}
*/
/*
		if (num < secondTrainingPhase) {
			task  = 0;
		}
		else {
			if (trainingItem == 2) {task = 1;}
			else {task = 0;}
		}
*/
		if (trainingItem == 2) {task = 1;}
			else {task = 0;}
		
		//cout << task << "---" << trainingItem << ": " << objectViewed << ": " << graspSeq << endl;
		//cout << "Object:  " << objectViewed << endl;

		if (num == firstTrainingPhase-1) {cout << "End of Training Phase 1" << endl; testJordan(1);}
		//if (num == secondTrainingPhase-1) {cout << "End of Training Phase 2" << endl; testJordan(2);}
		

		if (task == 0) {
			memcpy(expInstr, ExpInstr_Graps, sizeof(int)*2); 
			
			if (objectViewed == 0) {
				bbal++;
				memcpy(visionInput, bigBallVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/BBall/graspingNewNormSeq%i-BBall.data",graspSeq); //the name of the FILE string
				
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}
			}
			if (objectViewed == 1) {
				bbox++;
				memcpy(visionInput, bigBoxVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/BBox/graspingNewNormSeq%i-BBox.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 1\n" << std::endl; exit(1);}
			}
			if (objectViewed == 2) {
				sbal++;
				memcpy(visionInput, smallBallVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/SBall/graspingNewNormSeq%i-SBall.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 2\n" << std::endl; exit(1);}		
			}
			if (objectViewed == 3) {
				sbox++;
				memcpy(visionInput, smallBoxVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/SBox/graspingNewNormSeq%i-SBox.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 3\n" << std::endl; exit(1);}
			}
		}
		if (task == 1) { cout << "Categorise ";
			memcpy(expInstr, ExpInstr_Cat, sizeof(int)*2);
			
			if (objectViewed == 0) {
				bbal++;
				memcpy(visionInput, bigBallVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/SBox/graspingNewNormSeq%i-SBox.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}
			}
			if (objectViewed == 1) {
				bbox++;
				memcpy(visionInput, bigBoxVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/BBox/graspingNewNormSeq%i-BBox.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 1\n" << std::endl; exit(1);}
			}
			if (objectViewed == 2) {
				sbal++;				
				memcpy(visionInput, smallBallVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/SBall/graspingNewNormSeq%i-SBall.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 2\n" << std::endl; exit(1);}		
			}
			if (objectViewed == 3) {
				sbox++;				
				memcpy(visionInput, smallBoxVisionInput, sizeof(int)*100);
				char filename[100];
				filename[0]='\0';
				sprintf(filename,"data/1/BBall/graspingNewNormSeq%i-BBall.data",graspSeq); //the name of the FILE string
					
				// --- LOADING data from the file for training the network
				DATA = fopen(filename, "r" );
				if (!DATA) {std::cout << "can't open hand file for reading 3\n" << std::endl; exit(1);}
			}		
		}

		// --- READ the data from the file
		for (int line = 0; line < 10; line++) {	
			for (int input = 1; input <= 16; input++) {
				fscanf(DATA, "%lf", &val[input-1]); 
				targetOutput[line][input-1]= val[input-1];
				//cout << targetOutput[line][input-1] << " ";
				//if (input % 16 == 0) {cout<<endl;}
			}
		}

		// Set experimenter instruction as input for MT SOM		
		for (int j=0; j<MT_inputSize; j++) {
			MT_Data[j] = expInstr[j];
		}
		// Set visionInput as input for IT SOM
		for (int j=0; j<IT_inputSize; j++) {
			IT_Data[j] = visionInput[j];
		}	

		// IT SOM identifies the object in vision
		IT_input->setInputs(IT_Data);
		IT_net->step();
		
		RealVec IT_outputs(IT_m1->numNeurons());
		IT_outputs = IT_m1->outputs();
		
		// MT SOM identifies esperimenter instruction
		MT_input->setInputs(MT_Data);
		MT_net->step();
		
		RealVec MT_outputs(MT_m1->numNeurons());
		MT_outputs = MT_m1->outputs();

		// Take the outputs of IT & MT and feed it to PFC
		for (int j=0; j<PFC_inputSize; j++) {
			if (j < 100) {PFC_Data[j] = MT_outputs[j];}
			else {PFC_Data[j] = IT_outputs[j-100];}
		}
		
		PFC_input->setInputs(PFC_Data);
		PFC_net->step();	
		
		RealVec PFC_outputs(PFC_m1->numNeurons());
		PFC_outputs = PFC_m1->outputs();

		// Take the vision input and the PFC output to feed into the Jordan net
		for (int j=0; j<inputSize; j++) {
			if (j < 100) {
				if (trainingItem == 0) {Data[j] = 0;}
				else {Data[j] = PFC_outputs[j];}
			}
			else {Data[j] = visionInput[j-100];}
		}

		//cout << "Data" << endl;
		//for (int x=1; x<=200; x++) {
		//	cout << Data[x-1] << " ";
		//	if (x % 10 == 0) {cout<<endl;}
		//	if (x % 100 == 0) {cout<<endl;}
		//}

		// Take the vision input and the PFC output to feed into the Jordan net
		//for (int j=0; j<inputSize; j++) {
		//	if (j < 100) {Data[j] = PFC_outputs[j];}
		//	else {Data[j] = visionInput[j-100];}
		//}
	
		in->setInputs(Data);
/*
		Real weightsOutCat;
		Real weightsOut;
//		Realvec wOutCat(20);		
//		nnfw::saveXML("results/jordanSOMNet-before.xml",net);

		net->step();	

		for (int i=0; i<10; i++) {
			for (int j=0; j<2; j++) {
				weightsOutCat = l3->getWeight(i,j);
				//if (i<5) {wOutCat[i] = weightsOutCat;}
				//else {wOutCat[i+10] = weightsOutCat;}
				cout << weightsOutCat << "  ";	
			}
		}
		cout << endl;
		cout << endl;

		for (int i=0; i<10; i++) {
			for (int j=0; j<16; j++) {
				weightsOut = l4->getWeight(i,j);
				//if (i<5) {wOutCat[i] = weightsOutCat;}
				//else {wOutCat[i+10] = weightsOutCat;}
				cout << weightsOut << "  ";	
			}
		}
		cout << endl;
		cout << endl;
*/

		net->step();

		//RealVec tempCombo(cont->numNeurons());
		//RealVec temp(out->numNeurons());
		RealVec tempCat(outCat->numNeurons());
		RealVec targetCategorise(outCat->numNeurons());	

		//temp = out->outputs();
		tempCat = outCat->outputs();

		for (int i=0; i<outCat->numNeurons(); i++) {
			// Check whether the object is natural or an artefact
			if (objectViewed == 0 || objectViewed == 2) 
				targetCategorise[i] = naturalObject[i];
			else
				targetCategorise[i] = artefactObject[i];
		}

		// Copy outputs from the categorisation and motor neurons to context neurons
		//for (int i=0; i<cont->numNeurons(); i++) {
		//	if (i<2) {tempCombo[i]= tempCat[i];}
		//	else {tempCombo[i] = temp[i-2];}
		//}

		//cont->setInputs(tempCombo);

//		cout << targetCategorise << endl;
//		cout << tempCat << endl;

		bpCat->setTeachingInput(outCat,targetCategorise);
		bpCat->learn();
/*
		for (int i=0; i<10; i++) {
			for (int j=0; j<2; j++) {
				weightsOutCat = l3->getWeight(i,j);
				//if (i<5) {wOutCat[i] = weightsOutCat;}
				//else {wOutCat[i+10] = weightsOutCat;}
				cout << weightsOutCat << "  ";	
			}
		}
		cout << endl;
		cout << endl;	

		for (int i=0; i<10; i++) {
			for (int j=0; j<16; j++) {
				weightsOut = l4->getWeight(i,j);
				//if (i<5) {wOutCat[i] = weightsOutCat;}
				//else {wOutCat[i+10] = weightsOutCat;}
				cout << weightsOut << "  ";	
			}
		}
		cout << endl;
		cout << endl;
	
*/
//		cout << targetCategorise << endl;
//		cout << tempCat << endl;
//		error = bp->getError(outCat);

		// --- PRINT errors
		RealVec deltasCat(outCat->numNeurons());
		RealVec MSECat(outCat->numNeurons());
		MSECat.zeroing();	
		deltasCat.zeroing();
		deltasCat = targetCategorise;			
		deltasCat-=tempCat; // calculate delta (difference from output to target)			
		MSECat+=deltasCat.square(); // square the delta for one line of input
		RealVec RMSECat (outCat->numNeurons());
		float RMSEavCat = 0.0;

		for (int z = 0; z < (int)outCat->numNeurons(); z++) {
			RMSECat[z] = sqrt(MSECat[z]); // calculate sqrt of MeanSquareError for each output neuron
			RMSEavCat +=RMSECat[z]; // and add RMSE for each output neuron
		}
		RMSEavCat /=outCat->numNeurons(); // average the sqrt MeanSquareError of the output layer for one input
		fprintf(f1, "%f\n", RMSEavCat); 
	
		cont->resetInputs();
//		cout << out->outputs() << endl;
//		cout << cont->inputs() << endl;
//		cout << endl;

		// --- MAIN LOOP IN ORDER TO LEARN THE TASK 
		// For each object step the net 10 times --- representing 10 grasping sequences		
		for (int seq=0; seq<10; seq++) {	
			net->step();	
			RealVec temp(out->numNeurons());
			temp = out->outputs();

			// Copy outputs from the categorisation and motor neurons to context neurons
			//for (int i=0; i<cont->numNeurons(); i++) {
			//	if (i<2) {tempCombo[i]= tempCat[i];}
			//else {tempCombo[i] = temp[i-2];}
			//}

			//cont->setInputs(tempCombo);
//			cont->setInputs(temp);

		
			RealVec target(out->numNeurons());			

			for (int i=0; i<(int)out->numNeurons(); i++) 
				target[i] = targetOutput[seq][i];

			bp->setTeachingInput(out,target);
			bp->learn();
		
			// Calculate error
			if (num % 500 == 0) {			
				// --- PRINT errors
				RealVec deltas(out->numNeurons());
				RealVec MSE(out->numNeurons());
				MSE.zeroing();	
				deltas.zeroing();
				deltas = target;			
				deltas-=temp; // calculate delta (difference from output to target)			
				MSE+=deltas.square(); // square the delta for one line of input
				RealVec RMSE (out->numNeurons());
				float RMSEav = 0.0;
				float RMSEarm = 0.0; // error for the arm (first 8 DOF)			
				float RMSEhand = 0.0; // error for the hand (second 8 DOF)
				float RMSEprecision = 0.0; // error for thumb and indes fingers (5 DOF)
				float RMSEpower = 0.0; // error for middle ring and pinky fingers (2 DOF)

				for (int z = 0; z < (int)out->numNeurons(); z++) {
					RMSE[z] = sqrt(MSE[z]); // calculate sqrt of MeanSquareError for each output neuron
					RMSEav +=RMSE[z]; // and add RMSE for each output neuron
					
					if (z <= 7) {RMSEarm +=RMSE[z];} // and add RMSE for each output neuron of the arm
						if (z > 7) {
							RMSEhand +=RMSE[z];
							if (z < 13) {RMSEprecision +=RMSE[z];}
							else {RMSEpower +=RMSE[z];}
						}
				}
				RMSEav /=out->numNeurons(); // average the sqrt MeanSquareError of the output layer for one input
				RMSEarm /= 8;
				RMSEhand /= 8;
				RMSEprecision /= 5;
				RMSEpower /= 3;					
				
				fprintf(f, "%f %f %f %f %f\n", RMSEav, RMSEarm, RMSEhand, RMSEprecision, RMSEpower); 
				//fprintf(f, "%f\n", RMSEav); 			
				//cout << RMSEav << endl;	
			} 
		} //cout << endl;

		cont->resetInputs();
		fclose(DATA);
	}
	fclose(f);

	nnfw::saveXML("results/jordanSOMNet.xml",net);
	testJordan(2);	
	cout << "BBall: " << bbal << "\nBBox: " << bbox << "\nSBall: " << sbal << "\nSBox: " << sbox << endl;
	
	return 0;
}

void testJordan(int testPhase) {

	IT_net = loadXML("data/IT_SOM.xml");
	MT_net = loadXML("data/MT_SOM.xml");
	PFC_net = loadXML("data/PFC_SOM.xml");

	const ClusterVec& cl = IT_net->clusters();
	for( nnfw::u_int i=0; i<cl.size(); i++ ){                
    	 cl[i]->inputs().zeroing();
    	 cl[i]->outputs().zeroing();                                
 	}
	const ClusterVec& cl1 = MT_net->clusters();
	for( nnfw::u_int i=0; i<cl1.size(); i++ ){                
    	 cl1[i]->inputs().zeroing();
    	 cl1[i]->outputs().zeroing();                                
 	}
	const ClusterVec& cl2 = PFC_net->clusters();
	for( nnfw::u_int i=0; i<cl2.size(); i++ ){                
    	 cl2[i]->inputs().zeroing();
    	 cl2[i]->outputs().zeroing();                                
 	}

	IT_input = (SimpleCluster*)IT_net->getByName("IT_input");
	IT_m1 = (CompetitiveCluster*)IT_net->getByName("IT_map");
	IT_l1 = (NormLinker*)IT_net->getByName("IT_link");

	MT_input = (SimpleCluster*)MT_net->getByName("MT_input");
	MT_m1 = (CompetitiveCluster*)MT_net->getByName("MT_map");
	MT_l1 = (NormLinker*)MT_net->getByName("MT_link");
	
	PFC_input = (SimpleCluster*)PFC_net->getByName("PFC_input");
	PFC_m1 = (CompetitiveCluster*)PFC_net->getByName("PFC_map");
	PFC_l1 = (NormLinker*)PFC_net->getByName("PFC_link");
	
	RealVec Data = in->numNeurons();
	RealVec PFC_Data = PFC_input->numNeurons();
	RealVec IT_Data = IT_input->numNeurons();
	RealVec MT_Data = MT_input->numNeurons();

	// --- LOADING data from the file for testing the network
	FILE *DATA;

	char file1[100];
	file1[0]='\0';
	sprintf(file1,"results/testError%i.data",testPhase); //the name of the FILE string
	FILE *f = fopen(file1,"w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}

	char file2[100];
	file2[0]='\0';
	sprintf(file2,"results/testOutput%i.data",testPhase); //the name of the FILE string
	FILE *fout = fopen(file2,"w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}

	char file3[100];
	file3[0]='\0';
	sprintf(file3,"results/targets%i.data",testPhase); //the name of the FILE string
	FILE *ftarget = fopen(file3,"w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}



	int task = 0;	
	int objectViewed = 0;	
	int graspSeqTest[5] = {0,1,2,3,4}; //= 0;
	
	for (int e = 0; e < 2; e++) { // tasks
		for (int test = 0; test < 4; test++) { // objects
			for (int num = 0; num < 5; num++) { // grasping sequences
				
				//cout << task << "---" << objectViewed << ":" << graspSeqTest[num] << "\n";

				if (task == 0) {
					memcpy(expInstr, ExpInstr_Graps, sizeof(int)*2); 
					//cout << "Grasp:  " << endl;
					if (objectViewed == 0) {
						memcpy(visionInput, bigBallVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/BBall/graspingNewNormSeq%i-BBall.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}
					}
					if (objectViewed == 1) {
						memcpy(visionInput, bigBoxVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/BBox/graspingNewNormSeq%i-BBox.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 1\n" << std::endl; exit(1);}
					}
					if (objectViewed == 2) {
						memcpy(visionInput, smallBallVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/SBall/graspingNewNormSeq%i-SBall.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 2\n" << std::endl; exit(1);}		
					}
					if (objectViewed == 3) {
						memcpy(visionInput, smallBoxVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/SBox/graspingNewNormSeq%i-SBox.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 3\n" << std::endl; exit(1);}
					}
				}
				if (task == 1) {
					memcpy(expInstr, ExpInstr_Cat, sizeof(int)*2);
					//cout << "Categorise:  " << endl;
					if (objectViewed == 0) {
						memcpy(visionInput, bigBallVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/SBox/graspingNewNormSeq%i-SBox.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}
					}
					if (objectViewed == 1) {
						memcpy(visionInput, bigBoxVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/BBox/graspingNewNormSeq%i-BBox.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 1\n" << std::endl; exit(1);}
					}
					if (objectViewed == 2) {
						memcpy(visionInput, smallBallVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/SBall/graspingNewNormSeq%i-SBall.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 2\n" << std::endl; exit(1);}		
					}
					if (objectViewed == 3) {
						memcpy(visionInput, smallBoxVisionInput, sizeof(int)*100);
						char filename[100];
						filename[0]='\0';
						sprintf(filename,"data/1/BBall/graspingNewNormSeq%i-BBall.data",graspSeqTest[num]); //the name of the FILE string
						
						// --- LOADING data from the file for training the network
						DATA = fopen(filename, "r" );
						if (!DATA) {std::cout << "can't open hand file for reading 3\n" << std::endl; exit(1);}
					}		
				}

				// --- READ the data from the file
				for (int line = 0; line < 10; line++) {		
					for (int input = 1; input <= 16; input++) {
						fscanf(DATA, "%lf", &val[input-1]); 
						targetOutput[line][input-1]= val[input-1];	
						//cout << targetOutput[line][input-1] << " ";
						//if (input % 16 == 0) {cout<<endl;}
					}	
				}
			
				// Set experimenter instruction as input for MT SOM		
				for (int j=0; j<MT_inputSize; j++) {
					MT_Data[j] = expInstr[j];
				}
				// Set visionInput as input for IT SOM
				for (int j=0; j<IT_inputSize; j++) {
					IT_Data[j] = visionInput[j];
				}	

				IT_input->setInputs(IT_Data);
				IT_net->step();
				RealVec IT_outputs(IT_m1->numNeurons());
				IT_outputs = IT_m1->outputs();
		
				MT_input->setInputs(MT_Data);
				MT_net->step();
				RealVec MT_outputs(MT_m1->numNeurons());
				MT_outputs = MT_m1->outputs();

				// Take the outputs of IT & MT and feed it to PFC
				for (int j=0; j<PFC_inputSize; j++) {
					if (j < 100) {PFC_Data[j] = MT_outputs[j];}
					else {PFC_Data[j] = IT_outputs[j-100];}
				}
		
				PFC_input->setInputs(PFC_Data);
				PFC_net->step();	
				RealVec PFC_outputs(PFC_m1->numNeurons());
				PFC_outputs = PFC_m1->outputs();

				// Take the vision input and the PFC output to feed into the Jordan net
				for (int j=0; j<inputSize; j++) {
					if (j < 100) {Data[j] = PFC_outputs[j];}
					else {Data[j] = visionInput[j-100];}
				}
				in->setInputs(Data);		

				// --- PRINT visionInput 
				//for (int x=1; x<=100; x++) {
				//	cout << visionInput[x-1] << " ";
				//	if (x % 10 == 0) {cout<<endl;}
				//}

				// --- MAIN LOOP IN ORDER TO LEARN THE TASK 
				//for (int input=0; input<(int)in->numNeurons(); input++)
				//	in->setInput(input,visionInput[input]);

				net->step();
				RealVec outputsCat(outCat->numNeurons());
				outputsCat = outCat->outputs();
				cout << outputsCat << endl;

				for (int seq=0; seq<10; seq++) {		
					net->step();
					// --- COMPARE THE OUTPUT WITH THE TEACHING SET
					RealVec outputs(out->numNeurons());					outputs = out->outputs();							RealVec target(out->numNeurons());
	
					for (int i=0; i<out->numNeurons(); i++) 
						target[i] = targetOutput[seq][i];

					//if (test == 1 || test ==2) {cout << outputs << endl;}
					//cout << "Outputs" << endl;					
					//cout << outputs << endl;
					//cout << "Targets" << endl;						
					//cout << target << endl;

					//cout << cont->inputs() << endl;
					//cout << cont->outputs() << endl;
					//cout << out->outputs() << endl;
					//cout << endl;	
					//cout << hid->outputs() << endl;

					fprintf(fout, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8], outputs[9], outputs[10], outputs[11], outputs[12], outputs[13], outputs[14], outputs[15]); 

					fprintf(ftarget, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", target[0], target[1], target[2], target[3], target[4], target[5], target[6], target[7], target[8], target[9], target[10], target[11], target[12], target[13], target[14], target[15]); 

				
					// --- PRINT errors
					RealVec deltas(out->numNeurons());
					RealVec MSE(out->numNeurons());
					MSE.zeroing();	
					deltas.zeroing();
					deltas = target;
					deltas-=outputs; // calculate delta (difference from output to target)
					MSE+=deltas.square(); // square the delta for one line of input
					RealVec RMSE (out->numNeurons());
					float RMSEav = 0.0;	
					float RMSEarm = 0.0; // error for the arm (first 8 DOF)			
					float RMSEhand = 0.0; // error for the hand (second 8 DOF)
					float RMSEprecision = 0.0; // error for thumb and indes fingers (5 DOF)
					float RMSEpower = 0.0; // error for middle ring and pinky fingers (2 DOF)


					//cout << "Deltas\n" << deltas << endl; 
					//cout << "MSE\n" << MSE << endl; 
					//cout << "RMSE" << endl;;

					for (int z = 0; z < (int)out->numNeurons(); z++) {
						RMSE[z] = sqrt(MSE[z]); // calculate sqrt of MeanSquareError for each output neuron
						RMSEav +=RMSE[z]; // and add RMSE for each output neuron
						//cout << RMSE[z] << "\t";
					
						if (z <= 7) {RMSEarm +=RMSE[z];} // and add RMSE for each output neuron of the arm
						if (z > 7) {
							RMSEhand +=RMSE[z];
							if (z < 13) {RMSEprecision +=RMSE[z];}
							else {RMSEpower +=RMSE[z];}
						}

					}
					RMSEav /=out->numNeurons(); // average the sqrt MeanSquareError of the output layer for one input
					RMSEarm /= 8;
					RMSEhand /= 8;
					RMSEprecision /= 5;
					RMSEpower /= 3;	
					//cout << endl;
					//cout << "RMSEarm\n" << RMSEarm << "\n";
					//cout << "RMSEhand\n" << RMSEhand << "\n";
					//cout << "RMSEprecision\n" << RMSEprecision << "\n";
					//cout << "RMSEpower\n" << RMSEpower << "\n";

					//cout << RMSEav << endl;
					fprintf(f, "%f %f %f %f %f\n", RMSEav, RMSEarm, RMSEhand, RMSEprecision, RMSEpower); 
				} 	
				//cout << endl;
				cont->resetInputs();
			} 
			objectViewed++;	
		}
		task++;
		objectViewed = 0;
	}
	fclose(DATA); 
	fclose(f);
}
