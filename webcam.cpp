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

 Mat image;
 Mat channel[3];

 while (keepGoing) {

  image = cvQueryFrame(capture);
  imshow("webcam", image);

// thresholds on dark regions

  Mat gray, blurred_gray, threshold_gray;
  cvtColor(image, gray, CV_BGR2GRAY);
  blur(gray, blurred_gray, Size(width/10,height/20));
  equalizeHist(blurred_gray, blurred_gray);
  threshold(blurred_gray, threshold_gray, 40, 255, THRESH_BINARY_INV);
  imshow("threshold", threshold_gray);


  Moments lol = moments(threshold_gray, 1);
/*
  printf("m00: %f, m10: %f, m01: %f, m20: %f, m11: %f\n", lol.m00, lol.m10, lol.m01, lol.m20, lol.m11);
  printf("m02: %f, m30: %f, m21: %f, m12: %f, m03: %f\n", lol.m02, lol.m30, lol.m21, lol.m12, lol.m03);
  printf("mu20: %f, mu11: %f, mu02: %f, mu30: %f, mu21: %f, mu12: %f, mu03: %f\n", lol.mu20, lol.mu11, lol.mu02, lol.mu30, lol.mu21, lol.mu12, lol.mu03);

  printf("center (x,y) (%f,%f)\n",lol.m10/lol.m00,lol.m01/lol.m00);
*/
  circle(threshold_gray, Point(lol.m10/lol.m00,lol.m01/lol.m00),3,Scalar(128));
  imshow("threshold", threshold_gray);
  keepGoing = (waitKey(25)<0);

 }

 cvReleaseCapture(&capture);

 return 0;
}
