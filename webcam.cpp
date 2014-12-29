#include "webcam.hpp"
#include <chrono>

using namespace cv;
using namespace std;

Mat ellipticKernel(int width, int height = -1) {
 if (height==-1) {
  return getStructuringElement(MORPH_ELLIPSE,Size(width,width), Point(width/2, width/2));
 } else {
  return getStructuringElement(MORPH_ELLIPSE,Size(width,height), Point(width/2, height/2));
 }
}

unsigned long long getMilliseconds() {
 return chrono::system_clock::now().time_since_epoch()/chrono::milliseconds(1);
}

void morphFast(Mat inout, int smallsize = 100, int factor = 25, int eq = 1, int diler = 0) {
 int width, height;
 width = inout.size().width;
 height = inout.size().height;
 Mat downsample;
 resize(inout, downsample, Size(smallsize,smallsize));
 Mat kernel = ellipticKernel(factor);
 if (diler) {
  erode(downsample, downsample, kernel);
 } else {
  dilate(downsample, downsample, kernel);
 }
 if (eq) {
  equalizeHist(downsample, downsample);
 }
 resize(downsample, inout, Size(width, height));
}

int main (int argc, char** argv) {

 int tracker1, tracker2, tracker3;
 namedWindow("s",1);
 createTrackbar("1","s",&tracker1,100);
 createTrackbar("2","s",&tracker2,100);
 createTrackbar("3","s",&tracker3,100);

 CvCapture* capture = 0;
 int width, height, fps;

 capture = cvCaptureFromCAM(0);

 if (!capture) {
  printf("No camera detected!");
  return -1;
 }

 unsigned long long times[100];
 int f = 0;
 for (int i=0; i<100; i++)
  times[i] = 0;

 ifstream configFile (".config");

 if (configFile.is_open()) {

  //probably want to support corrupted .config
  string line;
  getline(configFile, line);
  istringstream(line)>>width;
  getline(configFile, line);
  istringstream(line)>>height;
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, width);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, height);

  configFile.close();

 } else {

  initResolutions();

  for (int i=36; i<150; i++) {
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, resolutions[i].width);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, resolutions[i].height);
  }

  width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
  height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

  ofstream configFileOut(".config");
  configFileOut << width;
  configFileOut << "\n";
  configFileOut << height;
  configFileOut << "\n";
  configFileOut.close();
 }

 bool keepGoing = true;

// srand(890);//not interested in good randomness

 for (int i = 0; i < 30; i++) {
  // capture some frames so exposure correction takes place
  cvQueryFrame(capture);
 }

 Mat background = cvQueryFrame(capture);
 background = background.clone();
 blur(background, background, Size(50,50));

 Mat image;
 Mat channel[3];
 unsigned long long timenow = getMilliseconds();

 CascadeClassifier mouth_cascade;
 mouth_cascade.load("Mouth.xml");

 while (keepGoing) {
  image = cvQueryFrame(capture);
  times[0] = getMilliseconds() - timenow;
  timenow = getMilliseconds();

// preprocess by rotating according to OVR roll
//  imshow("webcam", image);

// let's make multiple masks where 0=not mouth, 1=uncertain

// then multiply them together and multiply that with image
// and run haar classifier on image

  Mat gray, blurred_img;
  cvtColor(image, gray, CV_RGB2GRAY);
  blur(image, blurred_img, Size(50,50));
  times[1] = getMilliseconds() - timenow;
  timenow = getMilliseconds();

// this mask filters out areas which have not changed much
// this is horrible with new background; redo it
  Mat flow(height, width, CV_8UC1, 1);

  absdiff(blurred_img, background, flow);
  cvtColor(flow, flow, CV_RGB2GRAY);
  threshold(flow, flow, 6, 1, THRESH_BINARY);
  morphFast(flow);
  imshow("flow mask", gray.mul(flow));
  times[2] = getMilliseconds() - timenow;
  timenow = getMilliseconds();
//  flow = Mat(height, width, CV_8UC1, 1);


// this mask gets anything kind of dark (DK2) and dilates
  Mat kindofdark(height, width, CV_8UC1, 1);
  equalizeHist(gray, kindofdark);
  threshold(kindofdark, kindofdark, 100, 1, THRESH_BINARY_INV);
  morphFast(kindofdark, 100, 17, 0);
//  imshow("dark mask", gray.mul(kindofdark));
  times[3] = getMilliseconds() - timenow;
  timenow = getMilliseconds();

// combine mask with its opening
  Mat mask = flow.mul(kindofdark);
  Mat smallMask0, smallMask1;
  resize(mask, smallMask0, Size(width/5,height/5));
  Mat smallKernel = ellipticKernel(69,79);
  erode(smallMask0, smallMask1, smallKernel);
  dilate(smallMask1, smallMask1, smallKernel);
  bitwise_and(smallMask0, smallMask1, smallMask1);
  resize(smallMask1, mask, Size(width, height));
//  imshow("morph mask", gray.mul(mask));
  times[4] = getMilliseconds() - timenow;
  timenow = getMilliseconds();

// run haar classifier on nonflow parts of image
  vector<Rect> mouths;
  int scale = 3;
  Mat classifyThis;
  equalizeHist(gray, gray);
  resize(gray.mul(mask), classifyThis, Size(width/scale,height/scale));
  mouth_cascade.detectMultiScale(classifyThis, mouths, 1.1, 0, CV_HAAR_SCALE_IMAGE);
  Mat rectImage(height, width, CV_8UC1, Scalar(0));
  for (size_t i=0; i<mouths.size(); i++) {
   Rect scaled(mouths[i].x*scale, mouths[i].y*scale, mouths[i].width*scale,mouths[i].height*scale);
   Mat newRect(height, width, CV_8UC1, Scalar(0));
   rectangle(newRect, scaled, Scalar(1), CV_FILLED);
   rectImage += newRect;
  }
  double minVal, maxVal;//ignore minVal, it'll be 0
  minMaxLoc(rectImage, &minVal, &maxVal);
  Mat recThresh, recBinary;
  threshold(rectImage, recThresh, maxVal*0.8, 1, THRESH_BINARY);
// what's the point of this v ?
  threshold(rectImage, recBinary, 1, 1, THRESH_BINARY);
  bitwise_and(recBinary, mask, mask);
  times[5] = getMilliseconds() - timenow;
  timenow = getMilliseconds();
  imshow("mouth", recThresh.mul(gray));


/*
  Moments lol = moments(recThresh, 1);
  circle(image, Point(lol.m10/lol.m00,lol.m01/lol.m00),20,Scalar(128),30);
  imshow("leimage", image);
*/

// update background with new morph mask

  Mat mask_;
  subtract(1, mask ,mask_);
  Mat mask3, mask3_;
  channel[0] = mask;
  channel[1] = mask;
  channel[2] = mask;
  merge(channel, 3, mask3);
  channel[0] = mask_;
  channel[1] = mask_;
  channel[2] = mask_;
  merge(channel, 3, mask3_);

  background = background.mul(mask3) +
   (background.mul(mask3_)/2 + blurred_img.mul(mask3_)/2);
  times[6] = getMilliseconds() - timenow;
  timenow = getMilliseconds();

  imshow("bg", background);

  for (int i=0; i<7; i++) {
   printf("%llu , ", times[i]);
  }

  printf("\n");

  keepGoing = (waitKey(1)<0);

 }

 cvReleaseCapture(&capture);

 return 0;
}
