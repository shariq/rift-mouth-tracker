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

 while (keepGoing) {
  image = cvQueryFrame(capture);
  times[0] += getMilliseconds() - timenow;
  timenow = getMilliseconds();
// preprocess by rotating according to OVR roll
//  imshow("webcam", image);

// let's make multiple masks where 0=not mouth, 1=uncertain

// then multiply them together and multiply that with image
// and run haar classifier on image

  Mat gray, blurred_img;
  cvtColor(image, gray, CV_RGB2GRAY);
  blur(image, blurred_img, Size(50,50));
  times[1] += getMilliseconds() - timenow;
  timenow = getMilliseconds();

// this mask filters out areas with too many edges
// removed for now; it didn't generalize well
/*
  Mat canny;
  Canny(gray, canny, 50, 50, 3);
  blur(canny, canny, Size(width/20,height/20));
  bitwise_not(canny, canny);
  threshold(canny, canny, 200, 1, THRESH_BINARY);
  blur(canny*255, canny, Size(width/10, height/10));
  threshold(canny, canny, 220, 1, THRESH_BINARY);
  imshow("canny mask", gray.mul(canny));
*/

// this mask filters out areas which have not changed much
// background needs to be updated when person is not in frame
// use OVR SDK to do this later
  Mat flow;
  absdiff(blurred_img, background, flow);
  cvtColor(flow, flow, CV_RGB2GRAY);
  morphFast(flow);
  threshold(flow, flow, 60, 1, THRESH_BINARY);
//  imshow("flow mask", gray.mul(flow));
  times[2] += getMilliseconds() - timenow;
  timenow = getMilliseconds();

// this mask gets anything kind of dark (DK2) and dilates
  Mat kindofdark;
  equalizeHist(gray, kindofdark);
  threshold(kindofdark, kindofdark, 100, 1, THRESH_BINARY_INV);
  morphFast(kindofdark, 100, 17, 0);
//  imshow("dark mask", gray.mul(kindofdark));
  times[3] += getMilliseconds() - timenow;
  timenow = getMilliseconds();

// this mask gets rid of anything far away from red stuff
// did not work well and was slow
/*
  Mat notlips;
  Mat channels[3];
  split(image, channels);
  channels[2].convertTo(notlips, CV_32FC1);
  divide(notlips, gray, notlips, 1, CV_32FC1);
  //equalistHist is horrible for a red background
  //equalizeHist(notlips, notlips);
  threshold(notlips, notlips, tracker3/30.0, 1, THRESH_BINARY);
  notlips.convertTo(notlips, CV_8UC1);
  imshow("lip mask", notlips*255);
  Mat otherMorph = notlips.clone();
  int tx = tracker1+1-(tracker1%2);
  if (tx<3) tx=1;
  if (tx>90) tx=91;
  morphFast(notlips, 100, tx, 0, 0);
  int ty = tracker2+1-(tracker2%2);
  if (ty<3) ty=1;
  if (ty>90) ty=91;
  morphFast(notlips, 100, ty, 0, 1);
  imshow("lips2", notlips.mul(gray));
  morphFast(otherMorph, 100, tx, 0, 1);
  morphFast(otherMorph, 100, tx, 0, 0);
  imshow("lips3", otherMorph.mul(gray));
  waitKey(1);
*/

  Mat mask = flow.mul(kindofdark);
// open the mask
  Mat smallMask0, smallMask1;
  resize(mask, smallMask0, Size(width/5,height/5));
  Mat smallKernel = ellipticKernel(69,79);
  erode(smallMask0, smallMask1, smallKernel);
  dilate(smallMask1, smallMask1, smallKernel);
  bitwise_and(smallMask0, smallMask1, smallMask1);
  resize(smallMask1, mask, Size(width, height));
//  imshow("morph mask", gray.mul(mask));
  times[4] += getMilliseconds() - timenow;
  timenow = getMilliseconds();



// update background with new morph mask
// average what we know is background with prior background
// dilate it first since we really want to be sure it's bg

/*
// actually dilation is slow and our current mask is already
// really nice :)
  Mat dilatedMask;
  dilate(smallMask1, dilatedMask, smallKernel);
  resize(dilatedMask, dilatedMask, Size(width, height));
  imshow("erosion", dilatedMask.mul(gray));
*/
//  imshow("background", background);

/*
  Moments lol = moments(gray, 1);
  circle(image, Point(lol.m10/lol.m00,lol.m01/lol.m00),20,Scalar(128),30);
  imshow("leimage", image);
*/

  CascadeClassifier mouth_cascade;
  mouth_cascade.load("Mouth.xml");
  vector<Rect> mouths;
  int scale = 3;
  Mat classifyThis;
  equalizeHist(gray, gray);//ew; watch out not to use this later
  resize(gray.mul(mask), classifyThis, Size(width/scale,height/scale));
//  bilateralFilter(gray, classifyThis, 15, 10, 1);
  mouth_cascade.detectMultiScale(classifyThis, mouths, 1.1, 0, CV_HAAR_SCALE_IMAGE);
  Mat rectImage(height, width, CV_8UC1, Scalar(0));
  for (size_t i=0; i<mouths.size(); i++) {
   Rect scaled(mouths[i].x*scale, mouths[i].y*scale, mouths[i].width*scale,mouths[i].height*scale);
   Mat newRect(height, width, CV_8UC1, Scalar(0));
   rectangle(newRect, scaled, Scalar(1), CV_FILLED);
   rectImage += newRect;
  }
  threshold(rectImage, rectImage, tracker1, 1, THRESH_BINARY);
  times[6] += getMilliseconds() - timenow;
  timenow = getMilliseconds();
  imshow("MOUTH", rectImage.mul(gray));

  bitwise_and(rectImage, mask, mask);

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
  times[5] += getMilliseconds() - timenow;
  timenow = getMilliseconds();

  for (int i=0; i<7; i++) {
   printf("%llu , ", times[i]);
   times[i] = 0;
  }

  printf("\n");

  keepGoing = (waitKey(1)<0);


 }

 cvReleaseCapture(&capture);

 return 0;
}
