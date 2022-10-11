/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:22
*/
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "YOLOv5Detector.h"
#include "dataType.h"




int main(int argc, char *argv[])
{

    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
    detector->init(k_detect_model_path);

    std::vector<detect_result> results;
    cv::Mat frame = cv::imread("1.jpg");
    //Second/Millisecond/Microsecond  秒s/毫秒ms/微秒us
    auto start = std::chrono::system_clock::now();
    detector->detect(frame, results);
    auto end = std::chrono::system_clock::now();
    auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms

    detector->draw_frame(frame, results);

    cv::imshow("YOLOv5-6.x", frame);

    std::string output_file = cv::format("out%d.jpg", 1);
    cv::imwrite(output_file, frame);

    results.clear();





}
