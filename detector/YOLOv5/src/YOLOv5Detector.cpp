/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:41
*/
#include <YOLOv5Detector.h>
#include <iostream>

void YOLOv5Detector::init(std::string onnxpath)
{

    this->net = cv::dnn::readNetFromONNX(onnxpath);

    std::string file="./coco_80_labels_list.txt";
    std::ifstream ifs(file);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes_.push_back(line);
    }
}

void YOLOv5Detector::detect(cv::Mat & frame, std::vector<detect_result> &results)
{

    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));


    float x_factor = static_cast<float>(image.cols) / model_input_width_;
    float y_factor = static_cast<float>(image.rows) / model_input_height_;


    std::cout<<x_factor<<std::endl;
    std::cout<<y_factor<<std::endl;

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(model_input_width_, model_input_height_), cv::Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    cv::Mat preds = this->net.forward("output");//outputname


    cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++)
    {
        float box_conf = det_output.at<float>(i, 4);
        if (box_conf < nms_threshold_)
        {
            continue;
        }

        cv::Mat classes_confidences = det_output.row(i).colRange(5, 85);
        cv::Point classIdPoint;
        double cls_conf;
        cv::minMaxLoc(classes_confidences, 0, &cls_conf, 0, &classIdPoint);


        if (cls_conf > confidence_threshold_)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(cls_conf * box_conf);
        }
    }

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indexes);
    for (size_t i = 0; i < indexes.size(); i++)
    {
        detect_result dr;
        int index = indexes[i];
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.confidence = confidences[index];
        results.push_back(dr);
    }

}
void YOLOv5Detector::draw_frame(cv::Mat & frame, std::vector<detect_result> &results)
{

    for(auto dr : results)
    {

        cv::rectangle(frame, dr.box , cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(dr.box .tl().x, dr.box .tl().y - 20), cv::Point(dr.box .br().x, dr.box .tl().y), cv::Scalar(255, 0, 0), -1);

        std::string label = cv::format("%.2f", dr.confidence);
        label = classes_[dr.classId ] + ":" + label;

        cv::putText(frame, label, cv::Point(dr.box.x, dr.box.y + 6), 1, 2, cv::Scalar(0, 255, 0),2);

    }
}
