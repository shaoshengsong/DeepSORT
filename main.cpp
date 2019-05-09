// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/types_c.h>

#include <opencv2/bgsegm.hpp>
#include "./DeepAppearanceDescriptor/FeatureTensor.h"

#include "KalmanFilter/tracker.h"

const char* keys =
    "{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
    "{image i        |<none>| input image   }"
    "{video v       |<none>| input video   }"
    ;


// yolo parameter
// Initialize the parameters
const float confThreshold = 0.5; // Confidence threshold
const float nmsThreshold = 0.4;  // Non-maximum suppression threshold
const int inpWidth = 416;  // Width of network's input image
const int inpHeight = 416; // Height of network's input image
std::vector< std::string> classes;

//Deep SORT parameter

const int nn_budget=100;
const float max_cosine_distance=0.2;
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const  std::vector<cv::Mat>& out,   DETECTIONS& d);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d);
int main(int argc, char** argv)
{
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
  if (parser.has("help"))
    {
      parser.printMessage();
      return 0;
    }

  //deep SORT
  tracker mytracker(max_cosine_distance, nn_budget);
  //yolo
  // Load names of classes
  std::string classesFile = "coco.names";
  std::ifstream ifs(classesFile.c_str());
  std::string line;
  while (getline(ifs, line)) classes.push_back(line);

  // Give the configuration and weight files for the model
  cv::String modelConfiguration = "yolov3.cfg";
  cv::String modelWeights = "yolov3.weights";

  // Load the network
  cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // Open a video file or an image file or a camera stream.
  std::string str, outputFile;
  cv::VideoCapture cap;
  cv::VideoWriter video;
  cv::Mat frame, blob;

  try {

    outputFile = "yolo_out_cpp.avi";
    if (parser.has("image"))
      {
        // Open the image file
        str = parser.get<cv::String>("image");
        std::ifstream ifile(str);
        if (!ifile) throw("error");
        cap.open(str);
        str.replace(str.end()-4, str.end(), "_yolo_out_cpp.jpg");
        outputFile = str;
      }
    else if (parser.has("video"))
      {
        // Open the video file
        str = parser.get<cv::String>("video");
        std::ifstream ifile(str);
        if (!ifile) throw("error");
        cap.open(str);
        str.replace(str.end()-4, str.end(), "_yolo_out_cpp.avi");
        outputFile = str;
      }
    else
      {
        cap.open(0);
      }
    // Open the webcaom
    // else cap.open(parser.get<int>("device"));

  }
  catch(...) {
    std::cout << "Could not open the input image/video stream" <<  std::endl;
    return 0;
  }

  // Get the video writer initialized to save the output video
  if (!parser.has("image"))
    {
      video.open(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), 28.0,
                 cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
    }

  // Create a window
  static const  std::string kWinName = "Multiple Object Tracking";
  namedWindow(kWinName, cv::WINDOW_NORMAL);


  // Process frames.
  while (cv::waitKey(1) < 0)
    {
      // get frame from the video
      cap >> frame;

      // Stop the program if reached end of video
      if (frame.empty())
        {
          std::cout << "Done processing !!!" <<  std::endl;
          std::cout << "Output file is stored as " << outputFile <<  std::endl;
          cv::waitKey(3000);
          break;
        }
      // Create a 4D blob from a frame.
      cv::dnn::blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);

      //Sets the input to the network
      net.setInput(blob);

      // Runs the forward pass to get output of the output layers
      std::vector<cv::Mat> outs;
      net.forward(outs, getOutputsNames(net));

      // Remove the bounding boxes with low confidence
      DETECTIONS detections;
      postprocess(frame, outs,detections);

      std::cout<<"Detections size:"<<detections.size()<<std::endl;
      if(FeatureTensor::getInstance()->getRectsFeature(frame, detections))
        {
          std::cout << "Tensorflow get feature succeed!"<<std::endl;
          mytracker.predict();
          mytracker.update(detections);
          std::vector<RESULT_DATA> result;
          for(Track& track : mytracker.tracks) {
              if(!track.is_confirmed() || track.time_since_update > 1) continue;
              result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            }
          for(unsigned int k = 0; k < detections.size(); k++)
            {
              DETECTBOX tmpbox = detections[k].tlwh;
              cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
              cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);
              // cvScalar的储存顺序是B-G-R，CV_RGB的储存顺序是R-G-B

              for(unsigned int k = 0; k < result.size(); k++)
                {
                  DETECTBOX tmp = result[k].second;
                  cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                  rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

                  std::string label = cv::format("%d", result[k].first);
                  cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
                }
            }
        }
      else
        {
          std::cout << "Tensorflow get feature failed!"<<std::endl;;
        }
      // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
      std::vector<double> layersTimes;
      double freq = cv::getTickFrequency() / 1000;
      double t = net.getPerfProfile(layersTimes) / freq;
      std::string label = cv::format("Inference time for a frame : %.2f ms", t);
      putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

      // Write the frame with the detection boxes
      cv::Mat detectedFrame;
      frame.convertTo(detectedFrame, CV_8U);
      if (parser.has("image")) imwrite(outputFile, detectedFrame);
      else video.write(detectedFrame);

      imshow(kWinName, frame);

    }

  cap.release();
  if (!parser.has("image")) video.release();

  return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const  std::vector<cv::Mat>& outs,DETECTIONS& d)
{
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (size_t i = 0; i < outs.size(); ++i)
    {
      // Scan through all the bounding boxes output from the network and keep only the
      // ones with high confidence scores. Assign the box's class label as the class
      // with the highest score for the box.
      float* data = (float*)outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
          cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
          cv::Point classIdPoint;
          double confidence;
          // Get the value and location of the maximum score
          cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
          if ( static_cast<float>(confidence) >(confThreshold))
            {
              int centerX = (int)(data[0] * frame.cols);
              int centerY = (int)(data[1] * frame.rows);
              int width = (int)(data[2] * frame.cols);
              int height = (int)(data[3] * frame.rows);
              int left = centerX - width / 2;
              int top = centerY - height / 2;

              classIds.push_back(classIdPoint.x);
              confidences.push_back((float)confidence);
              boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

  // Perform non maximum suppression to eliminate redundant overlapping boxes with
  // lower confidences
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
  for (size_t i = 0; i < indices.size(); ++i)
    {
      size_t idx =static_cast<size_t>(indices[i]);
      cv::Rect box = boxes[idx];
      //目标检测 代码的可视化
      //drawPred(classIds[idx], confidences[idx], box.x, box.y,box.x + box.width, box.y + box.height, frame);

      get_detections(DETECTBOX(box.x, box.y,box.width,  box.height),confidences[idx],d);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
  //Draw a rectangle displaying the bounding box
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

  //Get the label for the class name and its confidence
  std::string label = cv::format("%.2f", conf);
  if (!classes.empty())
    {
      CV_Assert(classId < (int)classes.size());
      label = classes[classId] + ":" + label;
    }

  //Display the label at the top of the bounding box
  int baseLine;
  cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = cv::max(top, labelSize.height);
  cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
}

// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net)
{
  static  std::vector<cv::String> names;
  if (names.empty())
    {
      //Get the indices of the output layers, i.e. the layers with unconnected outputs
      std::vector<int> outLayers = net.getUnconnectedOutLayers();

      //get the names of all the layers in the network
      std::vector<cv::String> layersNames = net.getLayerNames();

      // Get the names of the output layers in names
      names.resize(outLayers.size());
      for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
  return names;
}

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
  DETECTION_ROW tmpRow;
  tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

  tmpRow.confidence = confidence;
  d.push_back(tmpRow);
}
