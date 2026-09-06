#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace dt {

// Single-frame detection result (xyxy format, image coordinates, floating point)
struct Detection {
    cv::Rect2f box;   // x1,y1,x2,y2
    float conf = 0.f;
    int cls = 0;
    int idx = 0;      // index within the original detection set
};

// Tracker output (current box + ID of an activated track)
struct TrackOutput {
    cv::Rect2f box;             // current-frame box (xyxy)
    int track_id = 0;
    float score = 0.f;
    int cls = 0;
    std::vector<cv::Point2f> trajectory; // history of centers (for drawing lines)
};

// Detection set fed to the tracker for one frame
struct FrameDetections {
    std::vector<cv::Vec4f> xyxy; // N x [x1,y1,x2,y2]
    std::vector<float> conf;
    std::vector<int> cls;
    int size() const { return static_cast<int>(xyxy.size()); }
};

} // namespace dt
