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
  //imshow("webcam", image);

// thresholds on dark regions


  Mat gray, blurred_gray, threshold_gray;
  cvtColor(image, gray, CV_BGR2GRAY);

  blur(gray, blurred_gray, Size(width/10,height/20));
  equalizeHist(blurred_gray, blurred_gray);
  bitwise_not(blurred_gray, blurred_gray);
  threshold(blurred_gray, threshold_gray, 210, 1, THRESH_BINARY);
  imshow("threshold", threshold_gray);

  Mat canny;
  Canny(gray, canny, 50, 50, 3);
  blur(canny, canny, Size(width/20,height/20));
  bitwise_not(canny, canny);
  threshold(canny, canny, 220, 1, THRESH_BINARY);
  imshow("canny", canny);

  Mat mask;
  bitwise_or(canny, threshold_gray, mask);
  Mat kernel = Mat::ones(15, 15, CV_8UC1);
  morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1,-1), 2);
  imshow("yo",gray.mul(mask));

/*
  bitwise_not(gray,gray);
  Mat mask = threshold_gray.mul(gray);
  imshow("mask", mask);

  Moments lol = moments(mask, 1);
  circle(image, Point(lol.m10/lol.m00,lol.m01/lol.m00),20,Scalar(128),30);
  imshow("leimage", image);
*/
  keepGoing = (waitKey(25)<0);


 }

 cvReleaseCapture(&capture);

 return 0;
}
