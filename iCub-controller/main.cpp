#include <stdio.h>
#include <iostream>
#include "cv.h"
#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/dev/all.h>
#include "math.h"
#include "highgui.h"
#include <ace/config.h>
#include <ctype.h>
#include <fstream>
#include "nnfw.h"
#include "nnfwfactory.h"
#include "biasedcluster.h"
#include "simplecluster.h"
#include "fakecluster.h"
#include "dotlinker.h"
#include "copylinker.h"
#include "normlinker.h"
#include "liboutputfunctions.h"
#include "libradialfunctions.h"
#include "random.h"
#include "time.h"
#include "types.h"
#include "ionnfw.h"
#include "propertized.h"


using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace yarp::dev;
using namespace std;
using namespace nnfw;

ImageOf<PixelRgb> *imgInR;

char robotName[10] = "/icubSim";

int width, height;

char portNameInRight[80];
char cameraNameRight[80];  //default

int visionInput[100];

double bigBallArea = 1052.75; // 1040.25; // 64-bit - 32-bit
double bigBoxArea = 640.0;
double smallBallArea = 198.5;
double smallBoxArea = 165.0;

CvMemStorage* storage = 0;

IplImage *image = 0;
BufferedPort<ImageOf<PixelRgb> > portInRight;

BiasedCluster *in, *hid, *outCat, *out;
SimpleCluster* cont;
DotLinker *l1, *l2, *l3, *l4;
CopyLinker* cl1;
BaseNeuralNet* net;

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


int expInstr[2];
int ExpInstr_Graps[2]= {0,1};
int ExpInstr_Cat[2]= {1,0};

int inputSize=200; // 200 outputs from vision + PFC SOM
int PFC_inputSize=200; // 200 outputs from IT+MT SOM
int IT_inputSize=100; // 2D map of visual input
int MT_inputSize=2; // tasks



void setPixelValue(IplImage* image, int x, int y, int channel, int value);


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



int main(int argc, char* argv[])
{
  	Network yarp;
	Bottle bot;	
	
	Port commandsHead;  	
	Port commandsRightArm;
	Port commandsTorso;

	commandsHead.open("/rpc/head");
	commandsRightArm.open("/rpc/rightArm");
	commandsTorso.open("/rpc/torso");
	
	Network::connect("/rpc/head", "/icubSim/head/rpc:i");	
	Network::connect("/rpc/rightArm", "/icubSim/right_arm/rpc:i");
	Network::connect("/rpc/torso", "/icubSim/torso/rpc:i");
	Network::connect("/icubSim/touch", "/touch");
    
	/********** Create Objects and place them on the table **********/

	BufferedPort<Bottle> worldPort;
	worldPort.open("/rpc/world");
	Network::connect("/rpc/world", "/icubSim/world");
	Bottle& objectBot = worldPort.prepare();	

	//int targetObject = 3; // 0 = BigBall; 1 = BigBox; 2 = SmallBall; 3 = SmallBox
	int targetObject;
	cout << "Enter the object you want to create, where:\n 0 = Big Ball\n 1 = Big Box\n 2 = Small Ball\n 3 = Small Box\n";
	cout << "Target Object: "; 
	cin >> targetObject;
	
	double radius, xlength, ylength, zlength;
	double xpos, ypos, zpos;
	double rcol = 0.9;
	double gcol = 0.0;
	double bcol = 0.0;

	objectBot.clear();	objectBot.addString("world");    objectBot.addString("mk");
	if (targetObject == 0) {objectBot.addString("ball"); radius = 0.04; xpos = -0.02; ypos = 0.555; zpos = 0.25;}
	if (targetObject == 1) {objectBot.addString("box"); xlength = 0.05, ylength = 0.05, zlength = 0.05; xpos = -0.02; ypos = 0.54; zpos = 0.25;}
	if (targetObject == 2) {objectBot.addString("ball"); radius = 0.02; xpos = -0.02; ypos = 0.535; zpos = 0.25;}
	if (targetObject == 3) {objectBot.addString("box"); xlength = 0.025, ylength = 0.025, zlength = 0.025; xpos = -0.02; ypos = 0.528; zpos = 0.25;}

	if (targetObject == 0 || targetObject == 2) {objectBot.addDouble(radius);}
	else { 
		objectBot.addDouble(xlength);
		objectBot.addDouble(ylength);
		objectBot.addDouble(zlength);
	}

	objectBot.addDouble(xpos);
	objectBot.addDouble(ypos);
	objectBot.addDouble(zpos);
	objectBot.addDouble(rcol);
	objectBot.addDouble(bcol);
	objectBot.addDouble(gcol);
	worldPort.write();
	Time::delay(0.5);

	/********** Move head and eyes to view the object **********/

	bot.clear();
	bot.addVocab(Vocab::encode("set"));
    bot.addVocab(Vocab::encode("pos"));
	bot.addInt(0);
	bot.addDouble(-34);
	commandsHead.write(bot);

	bot.clear();
	bot.addVocab(Vocab::encode("set"));
    bot.addVocab(Vocab::encode("pos"));
	bot.addInt(3);
	bot.addDouble(-35);
	commandsHead.write(bot);
	Time::delay(2.5);

	/********** Look at object with right eye and save the image **********/

	sprintf(portNameInRight, "/zoran/vision/right");
	portInRight.open(portNameInRight);
		
	sprintf(cameraNameRight,"%s/cam/right",robotName);	
	printf("Connecting %s to %s tcp\n",cameraNameRight,portNameInRight);
	Network::connect(cameraNameRight,portNameInRight);

	//get first RGB images to get width, height:
	imgInR = portInRight.read(); //blocking buffered
  	width = imgInR->width();
  	height = imgInR->height();
  	printf("\n\nReceived right camera dimensions: w:%d, h:%d\n\n",width,height);
  	
  	printf("processing...\n");
  	
	image = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U, 3 );
	cvCvtColor((IplImage*)imgInR->getIplImage(), image, CV_RGB2BGR);
  	
  	cvSaveImage("Vision.jpg", image);
	
	/********** Sobel filter the image and foveate the object representing it with 0s & 1s **********/
	
	cvNamedWindow("Sobel", 1);
	cvNamedWindow("Real Image", 1);
	cvMoveWindow("Real Image", 650, 200);
	cvNamedWindow("VISUAL INPUT", 1);
	cvNamedWindow("Cont", 1);

	IplImage* dx= cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_16S, 1);
    IplImage* dy= cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_16S, 1);
	IplImage* dest_dx = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);
	IplImage* dest_dy = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);
	
	IplImage* AverageR = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 3);
	IplImage* AverageG = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 3);
	IplImage* AverageB = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 3);
	
	IplImage* GreyAverageR = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);
	IplImage* GreyAverageG = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);
	IplImage* GreyAverageB = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);

	IplImage* AverageTotal = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 3);
	IplImage* AverageTotalGrey = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);
	IplImage* Temp = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 1);
    
    IplImage* RedImage = cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
    IplImage* GreenImage = cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_8U, 1);	
    IplImage* BlueImage = cvCreateImage( cvSize(image->width,image->height), IPL_DEPTH_8U, 1);		
    
   	cvSplit(image, BlueImage, GreenImage, RedImage, 0);

	IplImage* SobelViz = cvCreateImage(cvSize((image->width),(image->height)), IPL_DEPTH_8U, 1);
	
	cvSobel( RedImage, dx, 1, 0, 3);
    cvConvertScaleAbs( dx , dest_dx, 1, 0);
    cvSobel( RedImage, dy, 0, 1, 3);
    cvConvertScaleAbs( dy , dest_dy, 1, 0); 
  
	cvMerge(dest_dx,dest_dy,Temp,NULL,AverageR);
	cvCvtColor(AverageR, GreyAverageR, CV_RGB2GRAY);
	cvZero(dest_dx);cvZero(dest_dy);

	cvSobel( GreenImage, dx, 1, 0, 3);
    cvConvertScaleAbs( dx , dest_dx, 1, 0);
    cvSobel( GreenImage, dy, 0, 1, 3);
    cvConvertScaleAbs( dy , dest_dy, 1, 0); 
  
	cvMerge(dest_dx,dest_dy,Temp,NULL,AverageG);
	cvCvtColor(AverageG, GreyAverageG, CV_RGB2GRAY);
	cvZero(dest_dx);cvZero(dest_dy);

    cvConvertScaleAbs( dx , dest_dx, 1, 0);
    cvSobel( BlueImage, dy, 0, 1, 3);
    cvConvertScaleAbs( dy , dest_dy, 1, 0); 
  
	cvMerge(dest_dx,dest_dy,Temp,NULL,AverageB);
	cvCvtColor(AverageB, GreyAverageB, CV_RGB2GRAY);
	cvZero(dest_dx);cvZero(dest_dy);

	cvMerge(GreyAverageR,GreyAverageB,GreyAverageG,NULL,AverageTotal);
  	cvCvtColor(AverageTotal, AverageTotalGrey, CV_RGB2GRAY);


	// Split the image into a 2D map of 10x10 which is used as input for the neural net
   	
   	int w = 127; //100; //AverageTotalGrey->width;
   	int h = 95; //80; AverageTotalGrey->height;
   	 
   	// image from simulator is of size 320x240 pixels   	   	
	int numSquaresX = 10; 
   	int numSquaresY = 10;

   	int rowsInSquare = 5; // h/numSquaresY;
   	int columnsInSquare = 5; // w/numSquaresX;
   	
   	CvScalar c;
   	double sum;
   	double threshold = 18;
    			
	int iInc = 0;
	int jInc = 0;
	
	for (int x = 1; x < numSquaresX*numSquaresY+1; x++) {
		for (int i = h + iInc; i < (h + rowsInSquare + iInc); i++) {
			for (int j = w + jInc; j < (w + columnsInSquare + jInc); j++) {
					c = cvGetAt(AverageTotalGrey,i,j);
					sum += c.val[0];
			}
		}
				
		if (sum/(rowsInSquare*columnsInSquare) > threshold){
			for (int i = h + iInc; i < (h + rowsInSquare + iInc); i++) {
				for (int j = w+ jInc; j < (w + columnsInSquare + jInc); j++) {
					setPixelValue(SobelViz, j, i, 8, 255 );
					visionInput[x-1] = 1;
				}
			}	
		} 
		else {
			for (int i = h + iInc; i < (h + rowsInSquare + iInc); i++) {
				for (int j = w + jInc; j < (w + columnsInSquare + jInc); j++) {
					setPixelValue(SobelViz, j, i, 8, 50 );
					visionInput[x-1] = 0;
				}
			}	
		}
		
		sum = 0;
		jInc += columnsInSquare;
				
		if (x % numSquaresX == 0){
			iInc += rowsInSquare;
			jInc = 0;
		}
	}
	
	for (int x=1; x<=100; x++) {
		cout << visionInput[x-1];
		if (x % 10 == 0) {cout << endl;}
	}	

	cvShowImage("Sobel", AverageTotalGrey);
	cvShowImage("VISUAL INPUT", SobelViz);
	cvShowImage("Real Image", image);

	cvSaveImage("SobelViz.jpg", SobelViz);
	cvSaveImage("Sobel.jpg", AverageTotalGrey);
	
    //cvWaitKey(0);
    
    /********** Indentify the object in vision by calculating the area and perimeters of its countours **********/

	CvSeq* contours;	
	IplImage* clone_1 = 0;
    clone_1 = cvCloneImage(SobelViz);	

	// create memory storage that will contain all the dynamic data
    storage = cvCreateMemStorage(0);
	IplImage* dst = cvCreateImage(cvSize(image->width,image->height),  IPL_DEPTH_8U, 3);
	cvThreshold(SobelViz, clone_1, 50, 255, CV_THRESH_BINARY);
	cvFindContours(clone_1, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

	double area,perimeter;
	int i = 0;

	while(contours) {
    	i++;
		cvDrawContours(dst,contours,CV_RGB(100,100,255),CV_RGB(100,100,255),0,1,8);
		area += fabs(cvContourArea(contours,CV_WHOLE_SEQ));
		perimeter += fabs(cvArcLength(contours,CV_WHOLE_SEQ,-1));
        contours = contours->h_next;
    }
	
	area = area/i;
	perimeter = perimeter/i;
	cout << "Area is:  " << area << endl;
	cout << "Perimeter is:  " << perimeter << endl;

	// clear memory storage - reset free space position
    cvClearMemStorage( storage );	

	cvShowImage("Cont", dst);
	cvSaveImage("Cont.jpg", dst);
    cvWaitKey(0);

    
    /********** Reach the object so that it could be grasped **********/

	BufferedPort<Vector> targetPort;
	targetPort.open("/zoran/target/in");
	Network::connect("/zoran/target/out","/zoran/target/in");

	Property options;
	options.put("device", "remote_controlboard");
  	options.put("local", "/zoran/motor/client");
  	options.put("remote", "/icubSim/right_arm");
  	
	PolyDriver robotArm(options);
  	if (!robotArm.isValid()) {
    	printf("Cannot connect to robot arm\n");
   		return 1;
  	}

	IPositionControl *pos;
  	IVelocityControl *vel;
  	IEncoders *enc;
	double ENCODERS[16];

	robotArm.view(pos);
	robotArm.view(vel);
  	robotArm.view(enc);

	if (pos==NULL || vel==NULL || enc==NULL) {
    	printf("Cannot get interface to robot head\n");
    	robotArm.close();
    	return 1;
    }

	int jnts = 0;
	pos->getAxes(&jnts);
	Vector setpoints;
	Vector tmp;
 	setpoints.resize(jnts);
	tmp.resize(jnts);

	// Final position of the arm joints
	double sp, sr, sy, elbw, wpro, wptch, wy, fabd;
	// Final position of the finger joints
	double to, tp, td, ip, id, mp, md, p;

	// Reach the object --- hand is placed above the object
	double reachPos1[] = {90.0, 0.0, 0.0, 106.0, 90.0, -31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	pos->positionMove(reachPos1);
	Time::delay(1.5);

	double reachPos2[] = {-47.0, 0.0, 7.0, 90.0, 90.0, -31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos2);
	Time::delay(1.5);

	// Set the end joint positions according to the object currently being viewed
	if (area > 650.0) {
		sp = -47.0; sr = 0.0; sy = 7.0; elbw = 8.0; wpro = 90.0; wptch = -30.0; wy = -15.0; fabd = 0.0;
	} else {	
		sp = -46.0; sr = 4.0; sy = 7.0; elbw = 8.0; wpro = 90.0; wptch = -20.0; wy = -15.0; fabd = 0.0;
		
		// Initial position for grasping small objects
		//double posInit[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
		//pos->positionMove(posInit);
		//Time::delay(1.5);

		bot.clear();
		bot.addVocab(Vocab::encode("set"));
		bot.addVocab(Vocab::encode("pos"));
		bot.addInt(2);
		bot.addDouble(2.8);
		commandsTorso.write(bot);
		Time::delay(0.5);
	}

	// Initial position for grasping small objects
	double posInit[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(posInit);
	Time::delay(1.5);

	// Load SOMs
	IT_net = loadXML("data/IT_SOM.xml");
	MT_net = loadXML("data/MT_SOM.xml");
	PFC_net = loadXML("data/PFC_SOM.xml");

	IT_input = (SimpleCluster*)IT_net->getByName("IT_input");
	IT_m1 = (CompetitiveCluster*)IT_net->getByName("IT_map");
	IT_l1 = (NormLinker*)IT_net->getByName("IT_link");

	MT_input = (SimpleCluster*)MT_net->getByName("MT_input");
	MT_m1 = (CompetitiveCluster*)MT_net->getByName("MT_map");
	MT_l1 = (NormLinker*)MT_net->getByName("MT_link");
	
	PFC_input = (SimpleCluster*)PFC_net->getByName("PFC_input");
	PFC_m1 = (CompetitiveCluster*)PFC_net->getByName("PFC_map");
	PFC_l1 = (NormLinker*)PFC_net->getByName("PFC_link");

	const ClusterVec& cl = IT_net->clusters();
	for( nnfw::u_int i=0; i<cl.size(); i++ ){                
    	 cl[i]->inputs().zeroing();
    	 cl[i]->outputs().zeroing();                                
 	}
	const ClusterVec& cl0 = MT_net->clusters();
	for( nnfw::u_int i=0; i<cl0.size(); i++ ){                
    	 cl0[i]->inputs().zeroing();
    	 cl0[i]->outputs().zeroing();                                
 	}
	const ClusterVec& cl3 = PFC_net->clusters();
	for( nnfw::u_int i=0; i<cl3.size(); i++ ){                
    	 cl3[i]->inputs().zeroing();
    	 cl3[i]->outputs().zeroing();                                
 	}

	// --- LOAD THE NEURAL NET VIA XML 	net = loadXML("data/renormalised/jordanSOMNet.xml"); 

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
	//cl1 = (CopyLinker*)net->getByName("OutCat2Cont");	
	cl1 = (CopyLinker*)net->getByName("Out2Cont");	
	
	const ClusterVec& cl4 = net->clusters();
	for( nnfw::u_int i=0; i<cl4.size(); i++ ){                
    	 cl4[i]->inputs().zeroing();
    	 cl4[i]->outputs().zeroing();                                
 	}

	RealVec Data = in->numNeurons();
	RealVec PFC_Data = PFC_input->numNeurons();
	RealVec IT_Data = IT_input->numNeurons();
	RealVec MT_Data = MT_input->numNeurons();

	//int task = 1; // 0 = perform normal grasp; 1 = perform categorisation grasp (power for ball, precision for box)
	int task;
	cout << "Enter the instruction, where:\n 0 = grasp\n 1 = categorise\n";
	cout << "Instruction: ";	
	cin >> task;

	if (task == 0) {memcpy(expInstr, ExpInstr_Graps, sizeof(int)*2);}
	if (task==1) {memcpy(expInstr, ExpInstr_Cat, sizeof(int)*2);}

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

	//for (int input=0; input<(int)in->numNeurons(); input++) 
	//	in->setInput(input,visionInput[input]);

	cout << "Natural = 0, 1; Artefact = 1, 0" << endl;


	for (int seq = 0; seq < 10; seq++) {	
		//cout << cont->inputs() << endl;		
		net->step();
		
		RealVec encod(out->numNeurons());
		RealVec outputs(out->numNeurons());
		RealVec outputsNorm(out->numNeurons());		outputs = out->outputs();
		
		RealVec outputsCat(outCat->numNeurons());
		outputsCat = outCat->outputs();
		cout << outputsCat << endl;
		

		for (int i=0; i<16; i++) {
			// Normalise the outputs
			if (i < 8) {outputsNorm[i] = ((2 * outputs[i]) - 1)*100;}
			else {outputsNorm[i] = outputs[i]*100;}			
		}

		double positionGrasp[] = {outputsNorm[0], outputsNorm[1], outputsNorm[2], outputsNorm[3], outputsNorm[4], outputsNorm[5], outputsNorm[6], outputsNorm[7], outputsNorm[8], outputsNorm[9], outputsNorm[10], outputsNorm[11], outputsNorm[12], outputsNorm[13], outputsNorm[14], outputsNorm[15]};
		pos->positionMove(positionGrasp);
		Time::delay(0.1);
		/*
		enc->getEncoders(ENCODERS);
	
		for (int i=0; i<16; i++) {
			encod[i] = ((ENCODERS[i]/100)+1)/2;						
		}
		cont->setInputs(encod);
		*/
		// For grasping the object when only 8 outputs in neural net		
		//double positionChange[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, outputs[0]*100, outputs[1]*100, outputs[2]*100, outputs[3]*100, outputs[4]*100, outputs[5]*100, outputs[6]*100, outputs[7]*100};
		//pos->positionMove(positionChange);
		//Time::delay(0.1);
	} 	

	Time::delay(2.0);
	//enc->getEncoders(ENCODERS);
	//double positionLift[] = {ENCODERS[0]-30, ENCODERS[1], ENCODERS[2], ENCODERS[3], ENCODERS[4], ENCODERS[5], ENCODERS[6], ENCODERS[7], ENCODERS[8], ENCODERS[9], ENCODERS[10], ENCODERS[11], ENCODERS[12], ENCODERS[13], ENCODERS[14], ENCODERS[15]};
	//pos->positionMove(positionLift);

	bot.clear();
	bot.addVocab(Vocab::encode("set"));
	bot.addVocab(Vocab::encode("pos"));
	bot.addInt(0);
	bot.addDouble(-75.0);
	commandsRightArm.write(bot);
//	Time::delay(0.5);
	
	bot.clear();
	bot.addVocab(Vocab::encode("set"));
    bot.addVocab(Vocab::encode("pos"));
	bot.addInt(0);
	bot.addDouble(-20);
	commandsHead.write(bot);

	bot.clear();
	bot.addVocab(Vocab::encode("set"));
    bot.addVocab(Vocab::encode("pos"));
	bot.addInt(3);
	bot.addDouble(-15);
	commandsHead.write(bot);

	return 0;
}

void setPixelValue(IplImage* image, int x, int y, int channel, int value) {
    assert(channel >= 0);
    //assert(channel < image->nChannels);
	assert(x >= 0);
    assert(y >= 0);
	assert(x < image->width);
	assert(y < image->height);    
    (((uchar*)(image->imageData + image->widthStep*(y) )))[x * image->nChannels + channel] = (uchar)value;
}

