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

 Mat image;
 while (keepGoing) {

  image = cvQueryFrame(capture);

  //imshow("webcam", image);

//  Mat image2;
//  bilateralFilter(image, image2, 6,12,3);

  Mat channel[3];
  split(image, channel);
  Mat avg;
  avg = (channel[0] + channel[1] + channel[2])/3.0;
  equalizeHist(channel[0],channel[0]);
  equalizeHist(channel[1],channel[1]);
  equalizeHist(channel[2],channel[2]);
  pow(channel[0] - avg, 2.0, channel[0]);
  pow(channel[1] - avg, 2.0, channel[1]);
  pow(channel[2] - avg, 2.0, channel[2]);
  Mat gray;
  merge(channel, 3, gray);
  imshow("yo", gray);
/*
  //channel[0,1,2] -> b,g,r
  Mat gray;
  divide(channel[2], channel[1], gray, 50);
  //gaah = channel[2] - channel[1];
*/
//  Mat hsv;
//  cvtColor(image2, hsv, CV_BGR2HSV);
//  inRange(hsv, Scalar(0,0,0,0), Scalar(180,255,30,0), image);

//  cvtColor(hsv, image, CV_HSV2BGR);

/*
  Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  vector<Point2f> corners;
// maxCorners, qualityLevel, minDistance
// blockSize, useHarrisDetector, k
  goodFeaturesToTrack(gray, corners, 30, 0.01, 10, Mat(),\
   3, false, 0.04);

  for (int i=0; i<corners.size(); i++) {
   circle(image, corners[i], 4, Scalar(100,100,0), -1, 8, 0);
  }
*/

  imshow("webcam", image);

  keepGoing = (waitKey(25)<0);

 }

 cvReleaseCapture(&capture);
 cvDestroyWindow("webcam");

 return 0;
}
