#include "webcam.hpp"

using namespace cv;
using namespace std;

int main (int argc, char** argv) {

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
 imshow("background", background);

 Mat image;
 Mat channel[3];

 while (keepGoing) {

  image = cvQueryFrame(capture);
  //imshow("webcam", image);

  Mat gray;
  cvtColor(image, gray, CV_RGB2GRAY);

  Mat canny;
  Canny(gray, canny, 50, 50, 3);
  blur(canny, canny, Size(width/20,height/20));
  bitwise_not(canny, canny);
  threshold(canny, canny, 220, 1, THRESH_BINARY);
  imshow("canny1", canny*255);
  waitKey(1);

  int kwidth, kheight;
  if (width/2 > height) {
   kwidth = height/4;
   kheight = height/8;
  } else {
   int kwidth = width/8;
   int kheight = width/16;
  }

  kwidth += (1-(kwidth%2));//round up to nearest odd
  kheight += (1-(kheight%2));//round up to nearest odd
  Size kernelSize;
  kernelSize = Size(kwidth, kheight);
  Mat kernel = getStructuringElement(MORPH_ELLIPSE, kernelSize);

  morphologyEx(canny, canny, MORPH_OPEN, kernel);
  imshow("canny2", canny*255);
  waitKey(1);
  morphologyEx(canny, canny, MORPH_CLOSE, kernel);
  imshow("canny3", canny*255);
  waitKey(1);
  erode(canny, canny, kernel);
  imshow("canny4", canny*255);
  waitKey(1);

  Mat flow;
  blur(image, flow, Size(50,50));
  absdiff(flow, background, flow);
  cvtColor(flow, flow, CV_RGB2GRAY);
  blur(flow, flow, Size(50,50));
  equalizeHist(flow, flow);
  Mat mask;
  threshold(flow, mask, 170, 1, THRESH_BINARY);
  mask = mask.mul(canny);
  dilate(mask, mask, kernel); // maybe repeat some more
  imshow("FLOW", gray.mul(mask));

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
