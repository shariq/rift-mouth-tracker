#include "webcam.hpp"

using namespace cv;
using namespace std;

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
  imshow("webcam", image);

// let's make multiple masks where 0=not mouth, 1=uncertain

// then multiply them together and multiply that with image
// and run haar classifier on image

  Mat gray;
  cvtColor(image, gray, CV_RGB2GRAY);

// this mask filters out areas with too many edges
  Mat canny;
  Canny(gray, canny, 50, 50, 3);
  blur(canny, canny, Size(width/20,height/20));
  bitwise_not(canny, canny);
  threshold(canny, canny, 200, 1, THRESH_BINARY);
  blur(canny*255, canny, Size(width/10, height/10));
  threshold(canny, canny, 220, 1, THRESH_BINARY);
  imshow("canny mask", gray.mul(canny));

// this mask filters out areas which have not changed much
// background needs to be updated when person is not in frame
// use OVR SDK to do this later
  Mat flow;
  blur(image, flow, Size(50,50));
  absdiff(flow, background, flow);
  cvtColor(flow, flow, CV_RGB2GRAY);
//  blur(flow, flow, Size(tracker1+1,tracker1+1));
//  equalizeHist(flow, flow);
  Mat downsample;
  resize(flow, downsample, Size(250,250));
  Mat flowKernel = getStructuringElement(MORPH_ELLIPSE,
   Size(55,55)
  );
  dilate(downsample, downsample, flowKernel);
  resize(downsample, flow, Size(width, height));
  equalizeHist(flow, flow);
  threshold(flow, flow, 60, 1, THRESH_BINARY);
  imshow("flow mask", gray.mul(flow));

//  Moments lol = moments(mask, 1);
//  circle(image, Point(lol.m10/lol.m00,lol.m01/lol.m00),20,Scalar(128),30);
//  imshow("leimage", image);

/*
  CascadeClassifier mouth_cascade;
  mouth_cascade.load("Mouth.xml");
  vector<Rect> mouths;
  Mat classifyThis;
  bilateralFilter(gray, classifyThis, 15, 10, 1);
  equalizeHist(classifyThis, classifyThis);
  classifyThis = classifyThis.mul(mask);
  mouth_cascade.detectMultiScale(classifyThis, mouths, 1.1, 2, CV_HAAR_SCALE_IMAGE);
  for (size_t i=0; i<mouths.size(); i++) {
   Point center( mouths[i].x + mouths[i].width*0.5, mouths[i].y + mouths[i].height*0.5 );
   ellipse( image, center, Size( mouths[i].width*0.5, mouths[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
  }
  imshow("MOUTH", image);
*/
  keepGoing = (waitKey(25)<0);


 }

 cvReleaseCapture(&capture);

 return 0;
}
