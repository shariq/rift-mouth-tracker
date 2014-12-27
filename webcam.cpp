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

  Mat black, blurred;
  split(image, channel);
  merge(channel, 3, black);
  blur(black, blurred, Size(width/4.5,height/9));
  split(blurred, channel);
  black = (channel[0] + channel[1] + channel[2])/3.0;
  equalizeHist(black, black);
  bitwise_not(black,black);
  threshold(black, black, 220, 255, THRESH_BINARY);
  imshow("black", black);


  Mat stdev;
//  split(image, channel);
  equalizeHist(channel[0], channel[0]);
  equalizeHist(channel[1], channel[1]);
  equalizeHist(channel[2], channel[2]);
  stdev = (channel[0] + channel[1] + channel[2])/3.0;
  pow(channel[0] - stdev, 2.0, channel[0]);
  pow(channel[1] - stdev, 2.0, channel[1]);
  pow(channel[2] - stdev, 2.0, channel[2]);
  stdev = channel[0] + channel[1] + channel[2];
  stdev = (stdev/3.0);
  imshow("stdev", stdev);


/*
  split(image, channel);
  channel[0] = channel[0].mul(black);
  channel[1] = channel[1].mul(black);
  channel[2] = channel[2].mul(black);
  merge(channel, 3, image);
*/
//  imshow("yox", image);

//do some weird morphological closing thing
//  Mat channel[3];


/*
  Mat canny;
  Canny(image, canny, 0, 50);
  imshow("canny", canny);
*/

/*
  Mat fill = image.clone();
  Point seed(rand()%width, rand()%height);
  floodFill(fill, seed, Scalar(200,0,0), 0, Scalar(0,0,0), Scalar(25,25,25));
  imshow("fill", fill);
*/


  keepGoing = (waitKey(25)<0);

 }

 cvReleaseCapture(&capture);

 return 0;
}
