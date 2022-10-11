#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
class detect_result
{
public:
    int classId;
    float confidence;
    cv::Rect_<float> box;
};



class YOLOv5Detector
{
public:
    void init(std::string onnxpath);
    void detect(cv::Mat & frame, std::vector<detect_result> &result);
     std::vector<std::string> classes_;
     void draw_frame(cv::Mat & frame, std::vector<detect_result> &results);
private:

    cv::dnn::Net net;

    const float confidence_threshold_ =0.25f;
    const float nms_threshold_ = 0.4f;

    const int model_input_width_ = 640;
    const int model_input_height_ = 640;

};
