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

#include "FeatureTensor.h"
#include "BYTETracker.h" //bytetrack
#include "tracker.h"//deepsort
//Deep SORT parameter

const int nn_budget=100;
const float max_cosine_distance=0.2;

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}


void test_deepsort(cv::Mat& frame, std::vector<detect_result>& results,tracker& mytracker)
{
    std::vector<detect_result> objects;

    DETECTIONS detections;
    for (detect_result dr : results)
    {
        //cv::putText(frame, classes[dr.classId], cv::Point(dr.box.tl().x+10, dr.box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .8, cv::Scalar(0, 255, 0));
        if(dr.classId == 0) //person
        {
            objects.push_back(dr);
            cv::rectangle(frame, dr.box, cv::Scalar(255, 0, 0), 2);
            get_detections(DETECTBOX(dr.box.x, dr.box.y,dr.box.width,  dr.box.height),dr.confidence,  detections);
        }
    }

    std::cout<<"begin track"<<std::endl;
    if(FeatureTensor::getInstance()->getRectsFeature(frame, detections))
    {
        std::cout << "get feature succeed!"<<std::endl;
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
    std::cout<<"end track"<<std::endl;
}


void test_bytetrack(cv::Mat& frame, std::vector<detect_result>& results,BYTETracker& tracker)
{
    std::vector<detect_result> objects;


    for (detect_result dr : results)
    {

        if(dr.classId == 0) //person
        {
            objects.push_back(dr);
        }
    }


    std::vector<STrack> output_stracks = tracker.update(objects);

    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical)
        {
            cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
            cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                    0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        }
    }


}
int main(int argc, char *argv[])
{
    //deepsort
    tracker mytracker(max_cosine_distance, nn_budget);
    //bytetrack
    int fps=20;
    BYTETracker bytetracker(fps, 30);
    //-----------------------------------------------------------------------
    // 加载类别名称
    std::vector<std::string> classes;
    std::string file="./coco_80_labels_list.txt";
    std::ifstream ifs(file);
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    //-----------------------------------------------------------------------

    std::cout<<"classes:"<<classes.size();
    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());


    detector->init(k_detect_model_path);

    std::cout<<"begin read video"<<std::endl;
    cv::VideoCapture capture("./1.mp4");

    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }
    std::cout<<"end read video"<<std::endl;
    std::vector<detect_result> results;
    int num_frames = 0;

    cv::VideoWriter video("out.avi",cv::VideoWriter::fourcc('M','J','P','G'),10, cv::Size(1920,1080));

    while (true)
    {
        cv::Mat frame;


        if (!capture.read(frame)) // if not success, break loop
        {
            std::cout<<"\n Cannot read the video file. please check your video.\n";
            break;
        }

        num_frames ++;
        //Second/Millisecond/Microsecond  秒s/毫秒ms/微秒us
        auto start = std::chrono::system_clock::now();
        detector->detect(frame, results);
        auto end = std::chrono::system_clock::now();
        auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms
        std::cout<<classes.size()<<":"<<results.size()<<":"<<num_frames<<std::endl;


        //test_deepsort(frame, results,mytracker);
        test_bytetrack(frame, results,bytetracker);

        cv::imshow("YOLOv5-6.x", frame);

        video.write(frame);

        if(cv::waitKey(30) == 27) // Wait for 'esc' key press to exit
        {
            break;
        }

        results.clear();


    }
    capture.release();
    video.release();
    cv::destroyAllWindows();


}
