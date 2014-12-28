#include "webcam.hpp"

using namespace cv;
using namespace std;

Mat ellipticKernel(int width, int height = -1) {
 if (height==-1) {
  return getStructuringElement(MORPH_ELLIPSE,Size(width,width), Point(width/2, width/2));
 } else {
  return getStructuringElement(MORPH_ELLIPSE,Size(width,height), Point(width/2, height/2));
 }
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

 while (keepGoing) {

  image = cvQueryFrame(capture);
// preprocess by rotating according to OVR roll
//  imshow("webcam", image);

// let's make multiple masks where 0=not mouth, 1=uncertain

// then multiply them together and multiply that with image
// and run haar classifier on image

  Mat gray, blurred_img;
  cvtColor(image, gray, CV_RGB2GRAY);
  blur(image, blurred_img, Size(50,50));

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
  imshow("flow mask", gray.mul(flow));

// this mask gets anything kind of dark (DK2) and dilates
  Mat kindofdark;
  equalizeHist(gray, kindofdark);
  threshold(kindofdark, kindofdark, 100, 1, THRESH_BINARY_INV);
  morphFast(kindofdark, 100, 17, 0);
  imshow("dark mask", gray.mul(kindofdark));

  Mat mask = flow.mul(kindofdark);
// close the mask
/*
  Mat smallMask;
  resize(mask, smallMask, Size(150,150));
  int t1 = tracker1+1-(tracker1%2);
  if (t1>50) t1=51;
  if (t1<3) t1=3;
  int t2 = tracker2+1-(tracker2%2);
  if (t2>50) t2=51;
  if (t2<3) t2=3;
  Mat erodeKernel = ellipticKernel(t1,t2);
  erode(smallMask, smallMask, erodeKernel);
  Mat dilateKernel = ellipticKernel(t1,t2);
  dilate(smallMask, smallMask, dilateKernel);
  resize(smallMask, smallMask, Size(width, height));
  bitwise_and(smallMask,mask,mask);
*/
  imshow("morph mask", gray.mul(mask));

// update background with new morph mask
// average what we know is background with prior background
// erode it first since we really want to be sure it's bg

  erode(mask, mask, erodeKernel);
  Mat mask_;
  subtract(1,mask,mask_);
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
   (background.mul(mask3_) + blurred_img.mul(mask3_))/2;

  imshow("background", background);

//  Moments lol = moments(mask, 1);
//  circle(image, Point(lol.m10/lol.m00,lol.m01/lol.m00),20,Scalar(128),30);
//  imshow("leimage", image);


  CascadeClassifier mouth_cascade;
  mouth_cascade.load("Mouth.xml");
  vector<Rect> mouths;
  Mat classifyThis;
  blur(gray, classifyThis, Size(10,10));
//  bilateralFilter(gray, classifyThis, 15, 10, 1);
  equalizeHist(classifyThis, classifyThis);
  classifyThis = classifyThis.mul(mask);
  mouth_cascade.detectMultiScale(classifyThis, mouths, 1.1, 5, CV_HAAR_SCALE_IMAGE);
  for (size_t i=0; i<mouths.size(); i++) {
   Point center( mouths[i].x + mouths[i].width*0.5, mouths[i].y + mouths[i].height*0.5 );
   ellipse( image, center, Size( mouths[i].width*0.5, mouths[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
  }
  imshow("MOUTH", image);


  keepGoing = (waitKey(25)<0);


 }

 cvReleaseCapture(&capture);

 return 0;
}
