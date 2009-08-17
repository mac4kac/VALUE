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
#include "biasedcluster.h"
#include "simplecluster.h"
#include "dotlinker.h"
#include "copylinker.h"
#include "liboutputfunctions.h"

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

BiasedCluster *in, *hid, *out;
SimpleCluster* cont;
DotLinker *l1, *l2, *l3;
CopyLinker* cl1;
BaseNeuralNet* net;

void setPixelValue(IplImage* image, int x, int y, int channel, int value);


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
/*
	// Big ball
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("ball");	
	objectBot.addDouble(0.04);
	objectBot.addDouble(-0.02);
	//objectBot.addDouble(0.65);	
	objectBot.addDouble(0.555);
	objectBot.addDouble(0.25);
	objectBot.addDouble(0.9);
	objectBot.addDouble(0.0);
	objectBot.addDouble(0.0);
	worldPort.write();
	Time::delay(0.5);
*/

	// Big box
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("box");	
	objectBot.addDouble(0.05);
	objectBot.addDouble(0.05);
	objectBot.addDouble(0.05);	
	objectBot.addDouble(-0.02);
	//objectBot.addDouble(0.65);
	objectBot.addDouble(0.54);
	objectBot.addDouble(0.25);
	objectBot.addDouble(0.9);
	objectBot.addDouble(0.0);
	objectBot.addDouble(0.0);
	worldPort.write();
	Time::delay(0.5);	
/*
	// Small ball
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("ball");	
	objectBot.addDouble(0.02);	
	objectBot.addDouble(-0.02);
	objectBot.addDouble(0.535);
	objectBot.addDouble(0.25);
	objectBot.addDouble(0.9);
	objectBot.addDouble(0.0);
	objectBot.addDouble(0.0);
	worldPort.write();
	Time::delay(0.5);


	// Small box
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("box");	
	objectBot.addDouble(0.025);
	objectBot.addDouble(0.025);
	objectBot.addDouble(0.025);	
	objectBot.addDouble(-0.02);
	//objectBot.addDouble(0.65);	
	objectBot.addDouble(0.528);
	objectBot.addDouble(0.25);
	objectBot.addDouble(0.9);
	objectBot.addDouble(0.0);
	objectBot.addDouble(0.0);
	worldPort.write();
	Time::delay(0.5);
*/
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
	cvNamedWindow("SOBEL VIZ", 1);
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
	cvShowImage("SOBEL VIZ", SobelViz);
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

	//sp=-34.5; sr=13.0; sy=47.0; elbw=39.0; wpro=74.0; wptch=-17.0; wy=20.0; fabd=0.0;
	//to=80.0; tp=20.0; td=24.0; ip=52.0; id=34.0; mp=0.0; md=0.0; p=0.0;
/*
	// Torso bends down to grasp BigBox, SmallBall & SmallBox	
	bot.clear();
	bot.addVocab(Vocab::encode("set"));
	bot.addVocab(Vocab::encode("pos"));
	bot.addInt(2);
	bot.addDouble(3);
	commandsTorso.write(bot);
	Time::delay(0.5);

	// Grasp 1
	double reachPos3[] = {-45.5, 3.5, 7.0, 8.0, 90.0, -36.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos3);
	Time::delay(1.5);
	
	double reachPos4SBox[] = {-45.5, 3.5, 7.0, 8.0, 90.0, -36.0, -15.0, 0.0, 83.0, 6.0, 17.0, 63.0, 34.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos4SBox);
	Time::delay(1.5);	
	
	//double reachPos4SBall[] = {-47.5, 3.5, 7.0, 8.0, 90.0, -29.0, -15.0, 0.0, 64.0, 1.0, 36.0, 62.0, 31.0, 0.0, 0.0, 0.0};	
	//pos->positionMove(reachPos4SBall);
	//Time::delay(1.5);

	//double reachPos4BBox[] = {-47.0, 0.0, 7.0, 8.0, 90.0, -29.0, -15.0, 0.0, 62.0, 0.0, 43.0, 49.0, 32.0, 49.0, 35.0, 49.0};	
	//pos->positionMove(reachPos4BBox);
	//Time::delay(1.5);

	//double reachPos4BBall[] = {-47.0, 0.0, 7.0, 8.0, 90.0, -29.0, -15.0, 0.0, 53.0, 0.0, 33.0, 49.0, 30.0, 49.0, 31.0, 41.0};	
	//pos->positionMove(reachPos4BBall);
	//Time::delay(1.5);
	
	// Grasp 2
	double reachPos3[] = {-41.0, 14.0, 49.0, 25.0, 57.0, -23.0, -8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos3);
	Time::delay(1.5);

	double reachPos4SBox[] = {-41.0, 14.0, 49.0, 25.0, 57.0, -23.0, -8.0, 0.0, 81.0, 31.0, 14.0, 29.0, 52.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos4SBox);
	Time::delay(1.5);

	//double reachPos4SBall[] = {-41.0, 14.0, 49.0, 25.0, 57.0, -23.0, -8.0, 0.0, 52.0, 36.0, 28.0, 28.0, 47.0, 0.0, 0.0, 0.0};	
	//pos->positionMove(reachPos4SBall);
	//Time::delay(1.5);

	//double reachPos4BBox[] = {-41.0, 11.0, 49.0, 29.0, 57.0, -23.0, -8.0, 0.0, 64.0, 16.0, 44.0, 35.0, 39.0, 35.0, 37.0, 42.0};	
	//pos->positionMove(reachPos4BBox);
	//Time::delay(1.5);

	//double reachPos4BBall[] = {-41.0, 11.0, 49.0, 29.0, 57.0, -23.0, -8.0, 0.0, 56.0, 10.0, 34.0, 33.0, 32.0, 33.0, 32.0, 34.0};	
	//pos->positionMove(reachPos4BBall);
	//Time::delay(1.5);

	// Grasp 3
	double reachPos3[] = {-33.0, 9.0, 25.0, 33.0, 81.0, -25.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos3);
	Time::delay(1.5);

	double reachPos4SBox[] = {-33.0, 9.0, 25.0, 33.0, 81.0, -25.0, 1.0, 0.0, 89.0, 17.0, 11.0, 55.0, 32.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos4SBox);
	Time::delay(1.5);

	//double reachPos4SBall[] = {-33.0, 11.0, 30.0, 36.0, 84.0, -26.0, 0.0, 0.0, 62.0, 22.0, 33.0, 55.0, 24.0, 0.0, 0.0, 0.0};	
	//pos->positionMove(reachPos4SBall);
	//Time::delay(1.5);

	//double reachPos4BBox[] = {-36.0, 11.0, 35.0, 36.0, 73.0, -15.0, -18.0, 12.0, 86.0, 20.0, 34.0, 31.0, 48.0, 21.0, 39.0, 37.0};	
	//pos->positionMove(reachPos4BBox);
	//Time::delay(1.5);

	//double reachPos4BBall[] = {-36.0, 11.0, 35.0, 36.0, 73.0, -15.0, -18.0, 0.0, 75.0, 12.0, 22.0, 24.0, 35.0, 25.0, 33.0, 29.0};	
	//pos->positionMove(reachPos4BBall);
	//Time::delay(1.5);

	// Grasp 4
	double reachPos3[] = {-33.0, 3.0, 25.0, 34.0, 90.0, -17.0, 26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos3);
	Time::delay(1.5);

	double reachPos4SBox[] = {-33.0, 3.0, 25.0, 34.0, 90.0, -17.0, 26.0, 0.0, 68.0, 14.0, 33.0, 61.0, 34.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos4SBox);
	Time::delay(1.5);

	//double reachPos4SBall[] = {-33.0, 3.0, 25.0, 34.0, 77.0, -17.0, 26.0, 0.0, 74.0, 9.0, 31.0, 52.0, 32.0, 0.0, 0.0, 0.0};	
	//pos->positionMove(reachPos4SBall);
	//Time::delay(1.5);

	//double reachPos4BBox[] = {-34.0, 3.0, 25.0, 34.0, 77.0, -17.0, 3.0, 5.0, 87.0, 9.0, 25.0, 47.0, 41.0, 33.0, 36.0, 39.0};	
	//pos->positionMove(reachPos4BBox);
	//Time::delay(1.5);

	//double reachPos4BBall[] = {-34.0, 3.0, 25.0, 34.0, 77.0, -17.0, 3.0, 5.0, 73.0, 17.0, 21.0, 37.0, 35.0, 40.0, 28.0, 34.0};	
	//pos->positionMove(reachPos4BBall);
	//Time::delay(1.5);	

	// Grasp 5
	double reachPos3[] = {-34.5, 13.0, 47.0, 39.0, 76.0, -17.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos3);
	Time::delay(1.5);

	double reachPos4SBox[] = {-34.5, 13.0, 47.0, 39.0, 74.0, -17.0, 20.0, 0.0, 80.0, 20.0, 24.0, 52.0, 34.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos4SBox);
	Time::delay(1.5);

	//double reachPos4SBall[] = {-34.5, 13.0, 47.0, 39.0, 65.0, -17.0, 20.0, 0.0, 65.0, 26.0, 25.0, 47.0, 34.0, 0.0, 0.0, 0.0};	
	//pos->positionMove(reachPos4SBall);
	//Time::delay(1.5);

	//double reachPos4BBox[] = {-35.0, 13.0, 47.0, 39.0, 73.0, -24.0, 2.0, 3.0, 87.0, 11.0, 22.0, 36.0, 33.0, 19.0, 44.0, 39.0};	
	//pos->positionMove(reachPos4BBox);
	//Time::delay(1.5);

	//double reachPos4BBall[] = {-35.0, 13.0, 47.0, 39.0, 73.0, -24.0, 2.0, 3.0, 74.0, 8.0, 16.0, 33.0, 27.0, 31.0, 28.0, 32.0};	
	//pos->positionMove(reachPos4BBall);
	//Time::delay(1.5);
*/
/*
	// Grasping movement split into 10 steps
	double pos1[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-9*(to/10.0), tp-9*(tp/10.0), td-9*(td/10.0), ip-9*(ip/10.0), id-9*(id/10.0), mp-9*(mp/10.0), md-9*(md/10.0), p-9*(p/10.0)};	
	pos->positionMove(pos1);
	cout << (to-9*(to/10.0))/100 << " " << (tp-9*(tp/10.0))/100 << " " << (td-9*(td/10.0))/100 << " " << (ip-9*(ip/10.0))/100 << " " << (id-9*(id/10.0))/100 << " " << (mp-9*(mp/10.0))/100 << " " << (md-9*(md/10.0))/100 << " " << (p-9*(p/10.0))/100 << endl;	
	Time::delay(0.01);

	double pos2[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-8*(to/10.0), tp-8*(tp/10.0), td-8*(td/10.0), ip-8*(ip/10.0), id-8*(id/10.0), mp-8*(mp/10.0), md-8*(md/10.0), p-8*(p/10.0)};	
	pos->positionMove(pos2);
	cout << (to-8*(to/10.0))/100 << " " << (tp-8*(tp/10.0))/100 << " " << (td-8*(td/10.0))/100 << " " << (ip-8*(ip/10.0))/100 << " " << (id-8*(id/10.0))/100 << " " << (mp-8*(mp/10.0))/100 << " " << (md-8*(md/10.0))/100 << " " << (p-8*(p/10.0))/100 << endl;	
	Time::delay(0.01);

	double pos3[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-7*(to/10.0), tp-7*(tp/10.0), td-7*(td/10.0), ip-7*(ip/10.0), id-7*(id/10.0), mp-7*(mp/10.0), md-7*(md/10.0), p-7*(p/10.0)};	
	pos->positionMove(pos3);
	cout << (to-7*(to/10.0))/100 << " " << (tp-7*(tp/10.0))/100 << " " << (td-7*(td/10.0))/100 << " " << (ip-7*(ip/10.0))/100 << " " << (id-7*(id/10.0))/100 << " " << (mp-7*(mp/10.0))/100 << " " << (md-7*(md/10.0))/100 << " " << (p-7*(p/10.0))/100 << endl;	
	Time::delay(0.01);

	double pos4[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-6*(to/10.0), tp-6*(tp/10.0), td-6*(td/10.0), ip-6*(ip/10.0), id-6*(id/10.0), mp-6*(mp/10.0), md-6*(md/10.0), p-6*(p/10.0)};	
	pos->positionMove(pos4);
	cout << (to-6*(to/10.0))/100 << " " << (tp-6*(tp/10.0))/100 << " " << (td-6*(td/10.0))/100 << " " << (ip-6*(ip/10.0))/100 << " " << (id-6*(id/10.0))/100 << " " << (mp-6*(mp/10.0))/100 << " " << (md-6*(md/10.0))/100 << " " << (p-6*(p/10.0))/100 << endl;	
	Time::delay(0.01);

	double pos5[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-5*(to/10.0), tp-5*(tp/10.0), td-5*(td/10.0), ip-5*(ip/10.0), id-5*(id/10.0), mp-5*(mp/10.0), md-5*(md/10.0), p-5*(p/10.0)};	
	pos->positionMove(pos5);
	cout << (to-5*(to/10.0))/100 << " " << (tp-5*(tp/10.0))/100 << " " << (td-5*(td/10.0))/100 << " " << (ip-5*(ip/10.0))/100 << " " << (id-5*(id/10.0))/100 << " " << (mp-5*(mp/10.0))/100 << " " << (md-5*(md/10.0))/100 << " " << (p-5*(p/10.0))/100 << endl;	
	Time::delay(0.01);

	double pos6[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-4*(to/10.0), tp-4*(tp/10.0), td-4*(td/10.0), ip-4*(ip/10.0), id-4*(id/10.0), mp-4*(mp/10.0), md-4*(md/10.0), p-4*(p/10.0)};	
	pos->positionMove(pos6);
	cout << (to-4*(to/10.0))/100 << " " << (tp-4*(tp/10.0))/100 << " " << (td-4*(td/10.0))/100 << " " << (ip-4*(ip/10.0))/100 << " " << (id-4*(id/10.0))/100 << " " << (mp-4*(mp/10.0))/100 << " " << (md-4*(md/10.0))/100 << " " << (p-4*(p/10.0))/100 << endl;	
	Time::delay(0.01);

	double pos7[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-3*(to/10.0), tp-3*(tp/10.0), td-3*(td/10.0), ip-3*(ip/10.0), id-3*(id/10.0), mp-3*(mp/10.0), md-3*(md/10.0), p-3*(p/10.0)};	
	pos->positionMove(pos7);
	cout << (to-3*(to/10.0))/100 << " " << (tp-3*(tp/10.0))/100 << " " << (td-3*(td/10.0))/100 << " " << (ip-3*(ip/10.0))/100 << " " << (id-3*(id/10.0))/100 << " " << (mp-3*(mp/10.0))/100 << " " << (md-3*(md/10.0))/100 << " " << (p-3*(p/10.0))/100 << endl;
	Time::delay(0.01);

	double pos8[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-2*(to/10.0), tp-2*(tp/10.0), td-2*(td/10.0), ip-2*(ip/10.0), id-2*(id/10.0), mp-2*(mp/10.0), md-2*(md/10.0), p-2*(p/10.0)};
	pos->positionMove(pos8);
	cout << (to-2*(to/10.0))/100 << " " << (tp-2*(tp/10.0))/100 << " " << (td-2*(td/10.0))/100 << " " << (ip-2*(ip/10.0))/100 << " " << (id-2*(id/10.0))/100 << " " << (mp-2*(mp/10.0))/100 << " " << (md-2*(md/10.0))/100 << " " << (p-2*(p/10.0))/100 << endl;
	Time::delay(0.01);

	double pos9[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-(to/10.0), tp-(tp/10.0), td-(td/10.0), ip-(ip/10.0), id-(id/10.0), mp-(mp/10.0), md-(md/10.0), p-(p/10.0)};	
	pos->positionMove(pos9);
	cout << (to-(to/10.0))/100 << " " << (tp-(tp/10.0))/100 << " " << (td-(td/10.0))/100 << " " << (ip-(ip/10.0))/100 << " " << (id-(id/10.0))/100 << " " << (mp-(mp/10.0))/100 << " " << (md-(md/10.0))/100 << " " << (p-(p/10.0))/100 << endl;
	Time::delay(0.01);

	double pos10[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to, tp, td, ip, id, mp, md, p};	
	pos->positionMove(pos10);
	Time::delay(3.0);
	cout << to/100 << " " << tp/100 << " " << td/100 << " " << ip/100 << " " << id/100 << " " << mp/100 << " " << md/100 << " " << p/100 << endl;
	Time::delay(0.01);
*/

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

/*
	FILE *DATA;
	if (area == bigBallArea) {
		cout << "Seeing a big ball" << endl;
		DATA = fopen("data/graspBigBallOutputs.data", "r" );
		if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}			
	} 
	if (area == bigBoxArea) {
		cout << "Seeing a big box" << endl;
		DATA = fopen("data/graspBigBoxOutputs.data", "r" );
		if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}	
	}
	if (area == smallBallArea) {
		cout << "Seeing a small ball" << endl;
		DATA = fopen("data/graspSmallBallOutputs.data", "r" );
		if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}	
	}
	if (area == smallBoxArea) {
		cout << "Seeing a small box" << endl;
		DATA = fopen("data/graspSmallBoxOutputs.data", "r" );
		if (!DATA) {std::cout << "can't open hand file for reading 0\n" << std::endl; exit(1);}	
	}

	// --- READ the data from the file
	for (int line = 0; line < 10; line++) {		
		fscanf(DATA, "%lf %lf %lf %lf %lf %lf %lf %lf", &to, &tp, &td, &ip, &id, &mp, &md, &p); 
		double positionChange[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to*100, tp*100, td*100, ip*100, id*100, mp*100, md*100, p*100};
		pos->positionMove(positionChange);
		cout << to << "  " << tp << "  " << td << "  " << ip << "  " << id << "  " << mp << "  " << md << "  " << p << endl;	
		Time::delay(0.2);
	}

	fclose(DATA);
*/

	// --- LOAD THE NEURAL NET VIA XML 	//net = loadXML("data/jordanNet-8outputs.xml");
	net = loadXML("data/jordanNet2.xml"); //-16out-4o-4s-learn0.055.xml");

	// --- SET THE DIFFERENT LAYERS ACCORDINGLY
	in = (BiasedCluster*)net->getByName("Input");
	cont = (SimpleCluster*)net->getByName("Context");
	hid = (BiasedCluster*)net->getByName("Hidden");
	out = (BiasedCluster*)net->getByName("Output");
	
	l1 = (DotLinker*)net->getByName("In2Hid");
	l2 = (DotLinker*)net->getByName("Cont2Hid");
	l3 = (DotLinker*)net->getByName("Hid2Out");
	cl1 = (CopyLinker*)net->getByName("Out2Cont");	

	const ClusterVec& cl = net->clusters();
	for( nnfw::u_int i=0; i<cl.size(); i++ ){                
    	 cl[i]->inputs().zeroing();
    	 cl[i]->outputs().zeroing();                                
 	}

	for (int input=0; input<(int)in->numNeurons(); input++) 
		in->setInput(input,visionInput[input]);

	for (int seq = 0; seq < 10; seq++) {	
		//cout << cont->inputs() << endl;		
		net->step();
		
		RealVec encod(out->numNeurons());
		RealVec outputs(out->numNeurons());
		RealVec outputsNorm(out->numNeurons());		outputs = out->outputs();
		
		for (int i=0; i<16; i++) {
			// Normalise the outputs
			outputsNorm[i] = ((2 * outputs[i]) - 1)*100;	
			//outputsNorm[i] = outputs[i]*100;		
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

