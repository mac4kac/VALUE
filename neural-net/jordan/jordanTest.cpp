#include "nnfw.h"
#include "biasedcluster.h"
#include "simplecluster.h"
#include "dotlinker.h"
#include "copylinker.h"
#include "liboutputfunctions.h"
#include "backpropagationalgo.h"
#include "random.h"
#include "time.h"
#include <math.h>
#include "ionnfw.h"
#include <vector>
#include <string.h>

// 23 epochs for learning 2993 inputs = approx 70,000 iterations
#define numInput 2993 // defined in data/srn.data
#define numEpochs 23
#define numTestInput 32 // defined in data/predtest.data
#define learnRate 0.075

using namespace nnfw;
using namespace std;

//-------- Neural Network Structures
BiasedCluster *in, *hid, *out;
SimpleCluster* cont;
DotLinker *l1, *l2, *l3;
CopyLinker* cl1;
BaseNeuralNet* net;
BackPropagationAlgo* bp; 

double val[8]; // array for 8 inputs read from file
int visionInput[100]; // 2D map of the visual input
double targetOutput[10][8]; // 10 sequences for 8 joints in hand
int testInput[24] = {2,3,2,1,0,1,1,0,2,3,3,0,2,0,1,0,2,2,3,1,3,0,3,1}; // Test input for the network

int bigBallVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int bigBoxVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int smallBallVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int smallBoxVisionInput[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

void testJordan();

int main( int , char*[] ) {
 	
 	Random::setSeed(time(0));
    net = new BaseNeuralNet();
    
    // --- LOAD THE NEURAL NET VIA XML 
	net = loadXML( "net.xml" );
	// --- SET THE DIFFERENT LAYERS ACCORDINGLY
	in = (BiasedCluster*)net->getByName("Input");
	cont = (SimpleCluster*)net->getByName("Context");
	hid = (BiasedCluster*)net->getByName("Hidden");
	out = (BiasedCluster*)net->getByName("Output");
	
	l1 = (DotLinker*)net->getByName("In2Hid");
	l2 = (DotLinker*)net->getByName("Cont2Hid");
	l3 = (DotLinker*)net->getByName("Hid2Out");
	cl1 = (CopyLinker*)net->getByName("Out2Cont");	
	
	// --- RANDOMIZE THE NET
	net->randomize( -1.0, 1.0 );
	// --- REVERSE THE ORDER OF THE NET
	UpdatableVec rev_ord(net->order().size()); 
	rev_ord.assign_reverse(net->order());
	// --- CREATE THE BACKPROPAGATION ALGORITHM
	bp = new BackPropagationAlgo (net, rev_ord, learnRate);

	// --- PRINT PRECISION
	cout.precision( 5 );

	FILE *DATA;
	FILE *f = fopen("results/errorTraining.data","w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}
	int objectViewed;

	for (int num = 0; num < 60000; num++) {
		objectViewed = Random::flatInt(0,4);
		//objectViewed = 0;		
		//cout << objectViewed << endl;

		if (objectViewed == 0) {
			memcpy(visionInput, bigBallVisionInput, sizeof(int)*100);
			// --- LOADING data from the file for training the network
			DATA = fopen("data/graspBigBall.data", "r" );
			if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}
		}
		if (objectViewed == 1) {
			memcpy(visionInput, bigBoxVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspBigBox.data", "r" );
			if (!DATA) {std::cout << "can't open hand file for reading 1\n" << std::endl; exit(1);}
		}
		if (objectViewed == 2) {
			memcpy(visionInput, smallBallVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspSmallBall.data", "r" );
			if (!DATA) {std::cout << "can't open hand file for reading 2\n" << std::endl; exit(1);}		
		}
		if (objectViewed == 3) {
			memcpy(visionInput, smallBoxVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspSmallBox.data", "r" );
			if (!DATA) {std::cout << "can't open hand file for reading 3\n" << std::endl; exit(1);}
		}			
		// --- READ the data from the file
		for (int line = 0; line < 10; line++) {		
			for (int input = 0; input < 8; input++) {
				fscanf(DATA, "%lf", &val[input]); 
				targetOutput[line][input]= val[input];	
			}	
		}
/*
		// Print it 10 times for 10 grasping sequences
		//for (int i=0; i<10; i++) {
			for (int x=1; x<101; x++) {		
				cout << visionInput[x-1]; 
				if (x % 100 == 0) {cout << endl;}		
			}
		//}

		for (int l=0; l<10; l++) {
			for (int x=0; x<8; x++) {
				cout << targetOutput[l][x] << " ";
				//if (x % 10 == 0) {cout << endl;}
			}
			cout << endl;
			//cout << endl;
		}
*/
/*
		//for (int x=1; x<101; x++) {
		//	cout << visionInput[x-1] << " ";
		//	if (x % 10 == 0) {cout << endl;}		
		//}
		// Print it 10 times for 10 grasping sequences
		for (int i=0; i<10; i++) {
			for (int x=1; x<101; x++) {		
				fprintf(f, "%d ", visionInput[x-1]); 
				if (x % 100 == 0) {fprintf(f, "\n");}		
			}
		}
		// --- READ the data from the file
		for (int line = 0; line < 10; line++) {		
			for (int input = 0; input < 8; input++) {
				fscanf(DATA, "%lf", &val[input]); 
				targetOutput[line][input]= val[input];	
			}	
		}	
		for (int l=0; l<10; l++) {
			for (int x=0; x<8; x++) {
				cout << targetOutput[l][x] << " ";
				//if (x % 10 == 0) {cout << endl;}
			}
			cout << endl;
			//cout << endl;
		}
*/

/*		
	// --- LOADING data from the file for training the network
	FILE *DATA = fopen("data/graspBigBall.data", "r" );
	if (!DATA) {std::cout << "can't open hand file for reading\n" << std::endl; exit(1);}
	
	for (int line = 0; line < 10; line++) {
		// --- COPY visual input into inputLearnSet
		for (int x = 0; x < 100; x++) {	
			inputLearnSet[line][x] = bigBallVisionInput[x];	
		}
		// --- READ the data from the file
		for (int input = 0; input < 8; input++) {
			fscanf(DATA, "%lf", &val[input]); 
			inputLearnSet[line][input+100]= val[input];	
		}
	}	

	for (int l=0; l<10; l++) {
		for (int x=0; x<108; x++) {
			cout << inputLearnSet[l][x] << " ";
			//if (x % 10 == 0) {cout << endl;}
		}
		cout << endl;
		cout << endl;
	}

	for (int line = 0; line < numInput; line++){
		// --- READ the data from the file
		fscanf(DATA,"%lf %lf %lf  %lf\n",&val[0], &val[1], &val[2], &val[3]);
		for (int input = 0; input<(int)in->numNeurons(); input++) {		
			inputLearnSet[line][input]= val[input];
		//	cout << inputLearnSet[line][input] << " " ;
		}
		// cout << endl;
	}
*/
		// --- MAIN LOOP IN ORDER TO LEARN THE TASK 
		for (int input=0; input<(int)in->numNeurons(); input++) 
			in->setInput(input,visionInput[input]);

		// For each object step the net 10 times --- representing 10 grasping sequences		
		for (int seq=0; seq<10; seq++) {	
			net->step();	
			RealVec temp(out->numNeurons());
			temp = out->outputs();
			cont->setInputs(temp);
			RealVec target(out->numNeurons());			

			for (int i=0; i<(int)out->numNeurons(); i++) 
				target[i] = targetOutput[seq][i];
	
			bp->setTeachingInput(out,target);
			bp->learn();
		
			if (num % 1000 == 0) {			
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

				for (int z = 0; z < (int)out->numNeurons(); z++) {
					RMSE[z] = sqrt(MSE[z]); // calculate sqrt of MeanSquareError for each output neuron
					RMSEav +=RMSE[z]; // and add RMSE for each output neuron
				}
				RMSEav /=out->numNeurons(); // average the sqrt MeanSquareError of the output layer for one input
				fprintf(f, "%f\n", RMSEav); 			
				//cout << RMSEav << endl;	
			} 
		}
		cont->resetInputs();
		fclose(DATA);
	}
	fclose(f);
	nnfw::saveXML("results/jordanNet.xml",net);
	testJordan();			
	return 0;
}

void testJordan() {

	// --- LOADING data from the file for testing the network
	FILE *DATA;
	FILE *f = fopen("results/errorTest.data","w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}
	int objectViewed = 4;	
	
	for (int num = 0; num < 4; num++) {
		//objectViewed = Random::flatInt(0,4);
		objectViewed--; // = 0;//testInput[num];			
		cout << objectViewed << endl;

		if (objectViewed == 0) {
			memcpy(visionInput, bigBallVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspBigBall.data", "r" );
		}
		if (objectViewed == 1) {
			memcpy(visionInput, bigBoxVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspBigBox.data", "r" );
		}
		if (objectViewed == 2) {
			memcpy(visionInput, smallBallVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspSmallBall.data", "r" );
		}
		if (objectViewed == 3) {
			memcpy(visionInput, smallBoxVisionInput, sizeof(int)*100);
			DATA = fopen("data/graspSmallBox.data", "r" );
		}			
		// --- READ the data from the file
		for (int line = 0; line < 10; line++) {		
			for (int input = 0; input < 8; input++) {
				fscanf(DATA, "%lf", &val[input]); 
				targetOutput[line][input]= val[input];	
			}	
		}
		// --- MAIN LOOP IN ORDER TO LEARN THE TASK 
		for (int input=0; input<(int)in->numNeurons(); input++)
			in->setInput(input,visionInput[input]);

		for (int seq=0; seq<10; seq++) {		
			net->step();
			// --- COMPARE THE OUTPUT WITH THE TEACHING SET
			RealVec outputs(out->numNeurons());			outputs = out->outputs();					RealVec target(out->numNeurons());
	
			for (int i=0; i<out->numNeurons(); i++) 
				target[i] = targetOutput[seq][i];

			//cout << outputs << endl;
			//cout << target << endl;

			//cout << cont->inputs() << endl;
			//cout << cont->outputs() << endl;
			cout << out->outputs() << endl;
			//cout << endl;	

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

			for (int z = 0; z < (int)out->numNeurons(); z++) {
				RMSE[z] = sqrt(MSE[z]); // calculate sqrt of MeanSquareError for each output neuron
				RMSEav +=RMSE[z]; // and add RMSE for each output neuron
			}
			RMSEav /=out->numNeurons(); // average the sqrt MeanSquareError of the output layer for one input
			//cout << RMSEav << endl;
			fprintf(f, "%f\n", RMSEav); 
		} 	cout << endl;
			cont->resetInputs();
	}	
}
