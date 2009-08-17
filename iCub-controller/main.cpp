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

using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace yarp::dev;
using namespace std;

ImageOf<PixelRgb> *imgInR;
char robotName[10] = "/icubSim";

int width, height;

char portNameInRight[80];
char cameraNameRight[80];  //default

int visionInput[100];

double bigBallArea = 1040.25; //1052.75; // // 64-bit - 32-bit
double bigBoxArea = 640.0;
double smallBallArea = 198.5;
double smallBoxArea = 165.0;

CvMemStorage* storage = 0;

IplImage *image = 0;
BufferedPort<ImageOf<PixelRgb> > portInRight;

void setPixelValue(IplImage* image, int x, int y, int channel, int value);


int main(int argc, char* argv[])
{
  	Network yarp;
	Bottle bot;	
	Bottle *status;

	Port commandsHead;  	
	Port commandsRightArm;
	Port commandsTorso;
	BufferedPort<Bottle> touchPort; 
	commandsHead.open("/rpc/head");
	commandsRightArm.open("/rpc/rightArm");
	commandsTorso.open("/rpc/torso");
	touchPort.open("/touch");
	Network::connect("/rpc/head", "/icubSim/head/rpc:i");	
	Network::connect("/rpc/rightArm", "/icubSim/right_arm/rpc:i");
	Network::connect("/rpc/torso", "/icubSim/torso/rpc:i");
	Network::connect("/icubSim/touch", "/touch");
    
	/********** Create Objects and place them on the table **********/

	BufferedPort<Bottle> worldPort;
	worldPort.open("/rpc/world");
	Network::connect("/rpc/world", "/icubSim/world");
	Bottle& objectBot = worldPort.prepare();	

	// Big ball
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("sbal");	
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
/*

	// Big box
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("sbox");	
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

	// Small ball
	objectBot.clear();
	objectBot.addString("world");
    objectBot.addString("mk");
	objectBot.addString("sbal");	
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
	objectBot.addString("sbox");	
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
	Time::delay(7);

	/********** Look at object with right eye and save the image **********/

	//Network::init();
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
	//Network::fini();

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
		cout << visionInput[x-1] << ",";
		//if (x % 10 == 0) {cout << endl;}
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

		cout << "Area is:  " << area << endl;
		cout << "Perimeter is:  " << perimeter << endl;
			
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

	robotArm.view(pos);
  	robotArm.view(vel);
  	robotArm.view(enc);

	int jnts = 0;
	pos->getAxes(&jnts);
	Vector setpoints;
	Vector tmp;
 	setpoints.resize(jnts);
	tmp.resize(jnts);
	
	status = touchPort.read();
	int palmTouching = status->get(1).asInt();
	int thumbTouching = status->get(11).asInt();	
	int indexTouching = status->get(3).asInt();
	int middleTouching = status->get(5).asInt();
	int ringTouching = status->get(7).asInt();
	int pinkyTouching = status->get(9).asInt();

	// Reach the object --- hand is placed above the object
	double reachPos1[] = {90.0, 0.0, 0.0, 106.0, 90.0, -31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	pos->positionMove(reachPos1);
	Time::delay(1.5);
	
	double reachPos2[] = {-47.0, 0.0, 7.0, 90.0, 90.0, -31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos2);
	Time::delay(1.5);

	double reachPos3[] = {-47.0, 0.0, 7.0, 8.0, 90.0, -30.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
	pos->positionMove(reachPos3);
	Time::delay(1.5);
	//Time::delay(10.5);

	// Final position of the arm joints
	double sp, sr, sy, elbw, wpro, wptch, wy, fabd;
	// Final position of the finger joints
	double to, tp, td, ip, id, mp, md, p;
	
	// Set the end joint positions according to the object currently being viewed
	if (area > 600.0) {
	
		sp = -47.0; sr = 0.0; sy = 7.0; elbw = 8.0; wpro = 90.0; wptch = -30.0; wy = -15.0; fabd = 0.0;
	
		if (area == bigBallArea) {
			cout << "Seeing a big ball" << endl;
			to = 51.0; tp = 0.0; td = 31.0; ip = 48.01; id = 29.01; mp = 48.0; md = 30.0; p = 39.97;			
		} 
		else if (area == bigBoxArea) {
			cout << "Seeing a big box" << endl;
			to = 59.78; tp = 0.0; td = 40.78; ip = 40.05; id = 44.05; mp = 44.6; md = 43.6; p = 50.72;	
		}
	} 
	else {		
		sp = -46.0; sr = 4.0; sy = 7.0; elbw = 8.0; wpro = 90.0; wptch = -20.0; wy = -15.0; fabd = 0.0;
		
		// Initial position for grasping small objects
		double posInit[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};	
		pos->positionMove(posInit);
		Time::delay(1.5);

		bot.clear();
		bot.addVocab(Vocab::encode("set"));
		bot.addVocab(Vocab::encode("pos"));
		bot.addInt(2);
		bot.addDouble(3);
		commandsTorso.write(bot);
		Time::delay(0.5);
	
		if (area == smallBallArea) {
			cout << "Seeing a small ball" << endl;
			//to = 61.0; tp = 23.0; td = 41.0; ip = 11.0; id = 71.0; mp = 0.0; md = 0.0; p = 0.0; // for dynamic object
			to = 56.28; tp = 23.00; td = 36.28; ip = 8.69; id = 68.69; mp = 0.0; md = 0.0; p = 0.0; // for static object
		}
		else if (area == smallBoxArea) {
			cout << "Seeing a small box" << endl;
			//to = 50.0; tp = 37.0; td = 41.0; ip = 12.0; id = 82.0; mp = 0.0; md = 0.0; p = 0.0;	// for dynamic object	
			to = 48.82; tp = 37.00; td = 39.82; ip = 11.89; id = 81.89; mp = 0.0; md = 0.0; p = 0.0;	// for static object	
		}
	}
	
	cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << sp << "  " << sr << "  " << sy << "  " << elbw << "  " << wpro << "  " << wptch << "  " << wy << "  " << fabd << "  " << to << "  " << tp << "  " << td << "  " << ip << "  " << id << endl;

/*		
	//while (thumbTouching == 0 || indexTouching == 0) { // || middleTouching == 0 || ringTouching == 0|| pinkyTouching == 0) { //(status->get(3).asInt() != 1 & status->get(11).asInt() != 1) {
	while (indexTouching == 0) {
		if (thumbTouching != 1) {
			to += 0.01;
			//tp += 0.2;		
			td += 0.01;
		} 		

		if (indexTouching != 1) {
			ip += 0.01;	
			id += 0.01;
		}	
		if (middleTouching != 1) {
			mp += 0.01;
			md += 0.01;
		}
		if (ringTouching != 1 && pinkyTouching != 1) {
			p += 0.01;
		}	
				
		double positionChange[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to, tp, td, ip, id, mp, md, p};
		pos->positionMove(positionChange);
		//Time::delay(5.0);
		
		
		status = touchPort.read();
		thumbTouching = status->get(11).asInt();
		indexTouching = status->get(3).asInt();
		middleTouching = status->get(5).asInt();
		ringTouching = status->get(7).asInt();
		pinkyTouching = status->get(9).asInt();
		cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
		cout << to << "  " << tp << "  " << td << "  " << ip << "  " << id << "  " << mp << "  " << md << "  " << p << endl;	
		Time::delay(2.5);
	}
*/	
	
	
	// Grasping movement split into 10 steps
	double pos1[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-9*(to/10.0), tp-9*(tp/10.0), td-9*(td/10.0), ip-9*(ip/10.0), id-9*(id/10.0), mp-9*(mp/10.0), md-9*(md/10.0), p-9*(p/10.0)};	
	pos->positionMove(pos1);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << endl;
	cout << (to-9*(to/10.0))/100 << ", " << (tp-9*(tp/10.0))/100 << ", " << (td-9*(td/10.0))/100 << ", " << (ip-9*(ip/10.0))/100 << ", " << (id-9*(id/10.0))/100 << ", " << (mp-9*(mp/10.0))/100 << ", " << (md-9*(md/10.0))/100 << ", " << (p-9*(p/10.0))/100 << endl;	
	//Time::delay(2.0);

	double pos2[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-8*(to/10.0), tp-8*(tp/10.0), td-8*(td/10.0), ip-8*(ip/10.0), id-8*(id/10.0), mp-8*(mp/10.0), md-8*(md/10.0), p-8*(p/10.0)};	
	pos->positionMove(pos2);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-8*(to/10.0))/100 << ", " << (tp-8*(tp/10.0))/100 << ", " << (td-8*(td/10.0))/100 << ", " << (ip-8*(ip/10.0))/100 << ", " << (id-8*(id/10.0))/100 << ", " << (mp-8*(mp/10.0))/100 << ", " << (md-8*(md/10.0))/100 << ", " << (p-8*(p/10.0))/100 << endl;	
	//Time::delay(2.0);

	double pos3[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-7*(to/10.0), tp-7*(tp/10.0), td-7*(td/10.0), ip-7*(ip/10.0), id-7*(id/10.0), mp-7*(mp/10.0), md-7*(md/10.0), p-7*(p/10.0)};	
	pos->positionMove(pos3);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-7*(to/10.0))/100 << ", " << (tp-7*(tp/10.0))/100 << ", " << (td-7*(td/10.0))/100 << ", " << (ip-7*(ip/10.0))/100 << ", " << (id-7*(id/10.0))/100 << ", " << (mp-7*(mp/10.0))/100 << ", " << (md-7*(md/10.0))/100 << ", " << (p-7*(p/10.0))/100 << endl;	
	//Time::delay(2.0);

	double pos4[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-6*(to/10.0), tp-6*(tp/10.0), td-6*(td/10.0), ip-6*(ip/10.0), id-6*(id/10.0), mp-6*(mp/10.0), md-6*(md/10.0), p-6*(p/10.0)};	
	pos->positionMove(pos4);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-6*(to/10.0))/100 << ", " << (tp-6*(tp/10.0))/100 << ", " << (td-6*(td/10.0))/100 << ", " << (ip-6*(ip/10.0))/100 << ", " << (id-6*(id/10.0))/100 << ", " << (mp-6*(mp/10.0))/100 << ", " << (md-6*(md/10.0))/100 << ", " << (p-6*(p/10.0))/100 << endl;	
	//Time::delay(2.0);

	double pos5[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-5*(to/10.0), tp-5*(tp/10.0), td-5*(td/10.0), ip-5*(ip/10.0), id-5*(id/10.0), mp-5*(mp/10.0), md-5*(md/10.0), p-5*(p/10.0)};	
	pos->positionMove(pos5);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-5*(to/10.0))/100 << ", " << (tp-5*(tp/10.0))/100 << ", " << (td-5*(td/10.0))/100 << ", " << (ip-5*(ip/10.0))/100 << ", " << (id-5*(id/10.0))/100 << ", " << (mp-5*(mp/10.0))/100 << ", " << (md-5*(md/10.0))/100 << ", " << (p-5*(p/10.0))/100 << endl;	
	//Time::delay(2.0);

	double pos6[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-4*(to/10.0), tp-4*(tp/10.0), td-4*(td/10.0), ip-4*(ip/10.0), id-4*(id/10.0), mp-4*(mp/10.0), md-4*(md/10.0), p-4*(p/10.0)};	
	pos->positionMove(pos6);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-4*(to/10.0))/100 << ", " << (tp-4*(tp/10.0))/100 << ", " << (td-4*(td/10.0))/100 << ", " << (ip-4*(ip/10.0))/100 << ", " << (id-4*(id/10.0))/100 << ", " << (mp-4*(mp/10.0))/100 << ", " << (md-4*(md/10.0))/100 << ", " << (p-4*(p/10.0))/100 << endl;	
	//Time::delay(2.0);

	double pos7[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-3*(to/10.0), tp-3*(tp/10.0), td-3*(td/10.0), ip-3*(ip/10.0), id-3*(id/10.0), mp-3*(mp/10.0), md-3*(md/10.0), p-3*(p/10.0)};	
	pos->positionMove(pos7);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-3*(to/10.0))/100 << ", " << (tp-3*(tp/10.0))/100 << ", " << (td-3*(td/10.0))/100 << ", " << (ip-3*(ip/10.0))/100 << ", " << (id-3*(id/10.0))/100 << ", " << (mp-3*(mp/10.0))/100 << ", " << (md-3*(md/10.0))/100 << ", " << (p-3*(p/10.0))/100 << endl;
	//Time::delay(2.0);

	double pos8[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-2*(to/10.0), tp-2*(tp/10.0), td-2*(td/10.0), ip-2*(ip/10.0), id-2*(id/10.0), mp-2*(mp/10.0), md-2*(md/10.0), p-2*(p/10.0)};
	pos->positionMove(pos8);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-2*(to/10.0))/100 << ", " << (tp-2*(tp/10.0))/100 << ", " << (td-2*(td/10.0))/100 << ", " << (ip-2*(ip/10.0))/100 << ", " << (id-2*(id/10.0))/100 << ", " << (mp-2*(mp/10.0))/100 << ", " << (md-2*(md/10.0))/100 << ", " << (p-2*(p/10.0))/100 << endl;
	//Time::delay(2.0);

	double pos9[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to-(to/10.0), tp-(tp/10.0), td-(td/10.0), ip-(ip/10.0), id-(id/10.0), mp-(mp/10.0), md-(md/10.0), p-(p/10.0)};	
	pos->positionMove(pos9);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << (to-(to/10.0))/100 << ", " << (tp-(tp/10.0))/100 << ", " << (td-(td/10.0))/100 << ", " << (ip-(ip/10.0))/100 << ", " << (id-(id/10.0))/100 << ", " << (mp-(mp/10.0))/100 << ", " << (md-(md/10.0))/100 << ", " << (p-(p/10.0))/100 << endl;
	//Time::delay(2.0);

	double pos10[] = {sp, sr, sy, elbw, wpro, wptch, wy, fabd, to, tp, td, ip, id, mp, md, p};	
	pos->positionMove(pos10);
	Time::delay(3.0);
	status = touchPort.read();
	indexTouching = status->get(3).asInt();
	thumbTouching = status->get(11).asInt();
	//cout << thumbTouching << "  " << indexTouching << "  " << middleTouching << "  " << ringTouching << "  " << pinkyTouching << endl;
	cout << to/100 << ", " << tp/100 << ", " << td/100 << ", " << ip/100 << ", " << id/100 << ", " << mp/100 << ", " << md/100 << ", " << p/100 << endl;
	//Time::delay(2.0);

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
