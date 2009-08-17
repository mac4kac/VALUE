#include "types.h"
#include "nnfw.h"
#include "simplecluster.h"
#include "fakecluster.h"
#include "normlinker.h"
#include "liboutputfunctions.h"
#include "nnfwfactory.h"
#include "random.h"
#include "time.h"
#include "propertized.h"
#include "libradialfunctions.h"
#include <math.h>
#include "ionnfw.h"
#include <vector>
#include <SDL.h>
#include <stdio.h>

#define numIteration 2000

//for SDL
#define WIDTH 320
#define HEIGHT 240
#define BPP 4
#define DEPTH 32

using namespace nnfw;
using namespace std;

// Neural Network Structures
BaseNeuralNet* net;
SimpleCluster* input;
NormLinker* l1;

// Dimension of the SOM (10 neurons wide, 10 heigh)
int dim=10;

// Parameters for learning equations
float sigma0;
float tau1;
float tau2;
float eta0;
int nsteps;

// 100 input neurons Sobel processed image
int inputSize=100;

int visionInput[100]; // 2D map of the visual input

int VisionInput0[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int VisionInput1[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int VisionInput2[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int VisionInput3[100]= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

void learnSOM();
void saveSOMOutputs(int e, int l);
void testSOM();
//drawing function for the SOM
void drawrect(int x, int y, int w, int h, int color);
Uint32 tempColour; 
SDL_Surface *screen;
SDL_Event event;
SDL_Color colour;


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

CompetitiveCluster* m1;

int main( int , char*[]  ) {
	if( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
		fprintf(stderr,"Couldn't initialize SDL: %s\n", SDL_GetError());
		exit(1);
	}
	screen = SDL_SetVideoMode(WIDTH, HEIGHT, DEPTH, SDL_DOUBLEBUF | SDL_HWSURFACE);
	
	if ( screen == NULL ) {
		fprintf(stderr, "Couldn't set 640x480x8 video mode: %s\n",
			SDL_GetError());
		exit(1);
	}
	SDL_WM_SetCaption("Test for the SOM", "image");
	int h=0;
	atexit(SDL_Quit);
	
 	Random::setSeed(time(0));
 	
 	// Create input nodes 	
	input = new SimpleCluster(inputSize, "IT_input");
//	input -> setFunction(RampFunction(-100, 100, -100, +100));
	input -> setFunction(RampFunction(-100, 100, -100, +100));
	// Create SOM of dimension dim*dim
	m1 = new CompetitiveCluster(dim,dim,"IT_map");
	m1 -> setFunction(GaussFunction(0.0,0.5,1));
	// Link the input nodes to the SOM
	l1 = new NormLinker(input,m1,"IT_link");
  	net = new BaseNeuralNet();
	net -> addCluster(input);
	net -> addCluster(m1);
	net -> addLinker(l1);
	UpdatableVec ord;
	ord << input << l1 << m1;
	net -> setOrder(ord);
	// For each node in the SOM randomise the 3 weigths (RGB valuse)
	net -> randomize(-100,+100);
	nsteps = 0;
	//sigma0 = dim/2;
	tau2 = 1000;
	eta0 = 0.1f;
	//tau1 = 1000/log(sigma0);
	//tau1 = 1000/log(5);
		
	float sigmaBig = dim/2; // Large objects activate more neurons (neighbourhood is larger)
	float sigmaSmall = dim/3; //Smaller objects 

	RealVec Data = input->inputs();
			
	double Val[4][100];
		
	for (int e = 1; e <= numIteration; e++) { //numIteration; e++){	
		//cout << " numIteration: " << e << endl;
		
		int line  = Random::flatInt(0,4);
		
		if (line == 0) {memcpy(visionInput, VisionInput0, sizeof(int)*100); sigma0 = sigmaBig; tau1 = 1000/log(sigma0);} 

		if (line == 1) {memcpy(visionInput, VisionInput1, sizeof(int)*100); sigma0 = sigmaBig; tau1 = 1000/log(sigma0);}

		if (line == 2) {memcpy(visionInput, VisionInput2, sizeof(int)*100); sigma0 = sigmaSmall; tau1 = 1000/log(sigma0);}

		if (line == 3) {memcpy(visionInput, VisionInput3, sizeof(int)*100); sigma0 = sigmaSmall; tau1 = 1000/log(sigma0);}

		for (int j=0; j<inputSize; j++) {
			Data[j] = visionInput[j];
		}

		//--- PRINT visionInput 
		//for (int x=1; x<=100; x++) {
		//	cout << visionInput[x-1] << " ";
		//	if (x % 10 == 0) {cout<<endl;}
		//}
	
		input->setInputs(Data);// = Data;
		net->step();
		learnSOM();
		nsteps++;

		//if ( e % 500 == 0){saveSOMOutputs(e,line); cout<< line <<  endl;}

		int inc = -1;
		//keep the colour values for the SDL updates
		for (int y = 0; y < 10;  y++){
			for (int x = 0; x < 10; x++){
				inc++;
				Val[line][inc] = m1->outputs()[inc];
				//cout << Val[line][inc];
			}//cout << endl;
		}

		int vad = -1;
		if (e%10 == 0){
			for (int y = 0; y < 10;  y++){
				for (int x = 0; x < 10; x++){
					vad++;	
					if (Val[0][vad] * 255 > 60){
						
						colour.r = Val[0][vad] * 255;
						colour.g = 0;
						colour.b = 0;
						tempColour = SDL_MapRGB( screen->format, colour.r, colour.g, colour.b );
						drawrect(x*32, y*24, 32, 24, tempColour);
					}
					if (Val[1][vad] * 255 > 60){
					
						colour.r = 0;
						colour.g = Val[1][vad] * 255;
						colour.b = 0;
						tempColour = SDL_MapRGB( screen->format, colour.r, colour.g, colour.b );
						drawrect(x*32, y*24, 32, 24, tempColour);
					}
					if (Val[2][vad] * 255 > 60){
					
						colour.r = 0;
						colour.g = 0;
						colour.b = Val[2][vad] * 255;
						tempColour = SDL_MapRGB( screen->format, colour.r, colour.g, colour.b );
						drawrect(x*32, y*24, 32, 24, tempColour);
					}
					if (Val[3][vad] * 255 > 60){
					
						colour.r = Val[3][vad] * 255;
						colour.g = Val[3][vad] * 255;
						colour.b = 0;
						tempColour = SDL_MapRGB( screen->format, colour.r, colour.g, colour.b );
						drawrect(x*32, y*24, 32, 24, tempColour);
					}
				}
			}
		}
		//SDL_Delay(10);
		SDL_UpdateRect(screen, 0, 0, WIDTH, HEIGHT);
		
	}
	std::cout << "Press any key to exit : " << std::endl;
	
	while(1){
	if(SDL_PollEvent(&event)) {
		switch(event.type) {
	  		case SDL_KEYDOWN:
	  		exit(0);
			break;
		}
	}
	//testSOM();
	//nnfw::saveXML("results/IT_SOM.xml",net);    	
}
	return 0;	   
}

// Draw the rectangles in the SDL window
void drawrect(int x, int y, int w, int h, int color){
  SDL_Rect r;
  r.x = x;
  r.y = y;
  r.w = w;
  r.h = h;
  SDL_FillRect(screen, &r, color);
}

// Learning algorithm 
void learnSOM() { 
	float cx;
	float cy;
	m1 -> getCentre( cx, cy ); 
	
	for( int r=0; r<dim; r++ ) { // row dimentions
		for( int c=0; c<dim; c++ ) { // column dimensions
			float dist = (r-cx)*(r-cx) + (c-cy)*(c-cy); // find distance

			float sigman = pow( sigma0*exp(-nsteps/tau1), 2 ); // effecive width decrease with time
			float h = exp(-dist/(2*sigman)); // topological neighborhood function (gaussian)
			
			float etan = eta0*exp(-nsteps/tau2); // learning rate parameter
			
			if ( etan < 0.01 ) {
				etan=0.01f; // do not let it go below 0.01
			}
			int j = r*dim + c; // figure out the index inside the matrix
			RealMat& wmat = l1->matrix(); // matrix linker
			RealVec& tap = input->outputs();
			for( int i=0; i < inputSize; i++ ) {
				wmat[i][j] += etan*h*(tap[i] - wmat[i][j]); // synaptic adaptive process
			}
		}
	}
}

void saveSOMOutputs(int e, int line) {
	// NOW PRINT
	int v=0;
	char filename[100];
	filename[0]='\0';
	sprintf(filename,"results/SAveNEt%i_%i.txt",e, line); //the name of the FILE string
	FILE *f = fopen(filename,"w");
	if (!f) {printf("can't open config file for writing\n");exit(1);}
	
	for (int z =1; z <= 100; z++) {
		if (z%10 == 0) {								
		fprintf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", m1->outputs()[v + 0], m1->outputs()[v +1], m1->outputs()[v +2], m1->outputs()[v +3], m1->outputs()[v +4], m1->outputs()[v +5], m1->outputs()[v +6], m1->outputs()[v +7], m1->outputs()[v +8], m1->outputs()[v +9]);
		v = v + 10;
		//cout << v << endl;
		}	
	}
	fclose(f);
}

void testSOM() {

	int objectView;
	int objectsTest[4] = {0,1,2,3};
	RealVec Data = input->inputs();
	
	for (int test = 0; test < 4; test++) {
		objectView = objectsTest[test];
		cout << "Testing:\n" << objectView << endl;

		if (objectView == 0) {memcpy(visionInput, VisionInput0, sizeof(int)*100);} 

		if (objectView == 1) {memcpy(visionInput, VisionInput1, sizeof(int)*100);}

		if (objectView == 2) {memcpy(visionInput, VisionInput2, sizeof(int)*100);}

		if (objectView == 3) {memcpy(visionInput, VisionInput3, sizeof(int)*100);}

		for (int j=0; j<inputSize; j++) {
			Data[j] = visionInput[j];
		}
		
		input->setInputs(Data);
		net->step();
		saveSOMOutputs(objectView,test);
	}
}

