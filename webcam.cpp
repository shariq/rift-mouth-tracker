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

int main (int argc, char** argv) {

/*****
 begin camera setup
*****/

 CvCapture* capture = 0;
 int width, height;

 capture = cvCaptureFromCAM(0);

 if (!capture) {
  printf("No camera detected!");
  return -1;
 }

 ifstream config_file (".config");

 if (config_file.is_open()) {
// does not support corrupted .config

  string line;
  getline(config_file, line);
  istringstream(line)>>width;
  getline(config_file, line);
  istringstream(line)>>height;
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, width);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, height);

  config_file.close();

 } else {

  initResolutions();

  for (int i=36; i<150; i++) {
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, resolutions[i].width);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, resolutions[i].height);
  }

  width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
  height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

  ofstream config_file_out(".config");
  config_file_out << width;
  config_file_out << "\n";
  config_file_out << height;
  config_file_out << "\n";
  config_file_out.close();
 }

 for (int i = 0; i < 30; i++) {
  // capture some frames so exposure correction takes place
  cvQueryFrame(capture);
 }

/*****
 end camera setup
*****/

/*****
 filter setup
*****/

 Mat acbg(256, 256, CV_8UC3, Scalar(0,0,0));
// accumulated background
 Mat acbg_m(256, 256, CV_8UC1, Scalar(0));
// accumulated background mask

 Mat acfg(256, 256, CV_8UC3, Scalar(0,0,0));
// accumulated foreground
 Mat acfg_t(256, 256, CV_8UC3, Scalar(0));
// accumulated foreground threshold
 Mat defacbg(256, 256, CV_8UC3, Scalar(0,0,0));
 Mat iee(cvQueryFrame(capture));
 resize(iee, defacbg, Size(256,256));
 imshow("iee", iee);
/*****
 end filter setup
*****/

/*****
 misc
*****/

 unsigned long long times[100];
 for (int i=0; i<100; i++)
  times[i] = 0;

 int tracker1, tracker2, tracker3;
 namedWindow("s",1);
 createTrackbar("1","s",&tracker1,100);
 createTrackbar("2","s",&tracker2,100);
 createTrackbar("3","s",&tracker3,100);

 unsigned long long timenow = getMilliseconds();

 bool keep_going = true;

 CascadeClassifier mouth_cascade;
 mouth_cascade.load("Mouth.xml");

/*****
 begin filter loop
*****/

 while (keep_going) {

  Mat image(cvQueryFrame(capture));
//  imshow("webcam", image);
// at some point preprocess by rotating according to OVR roll
// right now really annoying because of state variables


  Mat gray, img_256, gray_256;
  resize(image, img_256, Size(256, 256));
  cvtColor(image, gray, CV_RGB2GRAY);
  cvtColor(img_256, gray_256, CV_RGB2GRAY);


// this mask gets anything kind of dark (DK2) and dilates
// should work *great*
  Mat gr_m;
// gray mask
  equalizeHist(gray_256, gr_m);
  threshold(gr_m, gr_m, 215, 1, THRESH_BINARY);
  dilate(gr_m, gr_m, ellipticKernel(23));
  erode(gr_m, gr_m, ellipticKernel(45));

  imshow("gray mask", gray_256.mul(1-gr_m));

  bitwise_or(acbg_m, gr_m, acbg_m);
  imshow("accumulated bg mask", gray_256.mul(1-acbg_m));

// this mask watches for flow against accumulated bg
  Mat fl_m;
// flow mask
  absdiff(img_256, iee, fl_m);
//  absdiff(img_256, acbg, fl_m);
  cvtColor(fl_m, fl_m, CV_BGR2GRAY);
//  fl_m = acbg_m.mul(fl_m);
  threshold(fl_m, fl_m, tracker3*3, 1, THRESH_BINARY);
  int t1 = tracker1+1 - (tracker1%2);
  int t2 = tracker2+1 - (tracker2%2);
  if (t1<3) t1 = 3;
  if (t1>90) t1 = 91;
  if (t2<3) t2 = 3;
  if (t2>90) t2 = 91;
  dilate(fl_m, fl_m, ellipticKernel(t1));
  erode(fl_m, fl_m, ellipticKernel(t2));
  imshow("flow mask", fl_m*255);

  Mat bg_m;
  bitwise_and(acbg_m, fl_m, bg_m);
  bitwise_or(gr_m, bg_m, bg_m);
// maybe do some morphological operations on bg_m?
// previously combined bg_m with its opening

// ugly way to compute:
// acbg = [acbg'.(1-bg_m)]+[((acbg'+img_256)/2).bg_m]
// find a nicer way later
// problem is masks are 1 channel while images are 3 channel

  Mat bg_m3;
  Mat tmp0[3];
  tmp0[0] = tmp0[1] = tmp0[2] = bg_m;
  merge(tmp0, 3, bg_m3);
//  imshow("bg_m3", bg_m3*255);
  acbg = acbg.mul(Scalar(1,1,1)-bg_m3) + (acbg/2+img_256/2).mul(bg_m3);
//  imshow("acbg", acbg);


//  imshow("bg mask", gray_256.mul(1-bg_m));

/*
 // do some stuff with foreground and so on here

  Mat haar_m;
// bitwise_and(1 - bg_m, fg_m, haar_m);
  haar_m = 1 - bg_m;

// run haar classifier
  int scale = width/300;
  if (scale < 1)
   scale = 1;
// can't use 256x256 since haar isn't stretch invariant

  Mat thingy;
  resize(gray, thingy, Size(width/scale, height/scale));
  equalizeHist(thingy, thingy);
///////////////////
// need to change this after foreground stuff gets written

  vector<Rect> mouth_rects;
  resize(haar_m, haar_m, Size(width/scale, height/scale));

  bitwise_and(haar_m, thingy, thingy);
/////////////////

  mouth_cascade.detectMultiScale(thingy, mouth_rects, 1.1, 0, CV_HAAR_SCALE_IMAGE);
  Mat rect_image(height, width, CV_8UC1, Scalar(0));

  for (size_t i=0; i<mouth_rects.size(); i++) {
   Rect scaled(mouth_rects[i].x*scale, mouth_rects[i].y*scale, mouth_rects[i].width*scale,mouth_rects[i].height*scale);
   Mat new_rect(height, width, CV_8UC1, Scalar(0));
   rectangle(new_rect, scaled, Scalar(1), CV_FILLED);
   rect_image += new_rect;
  }

  double min_val, max_val;
  minMaxLoc(rect_image, &min_val, &max_val);

// or maybe equalize? this whole thing needs to be rewritten
// with the new fg and temporal coherence ideas

  Mat rect_thresh;
  threshold(rect_image, rect_thresh, max_val*0.9, 1, THRESH_BINARY);
  imshow("mouth", rect_thresh.mul(gray));


/*
  Moments m = moments(rect_thresh, 1);
  circle(image, Point(m.m10/m.m00,m.m01/m.m00),20,Scalar(128),30);
  imshow("centroid", image);
*/

  keep_going = (waitKey(1)<0);

 }

 cvReleaseCapture(&capture);

 return 0;
}

