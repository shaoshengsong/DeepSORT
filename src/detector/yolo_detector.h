#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include "tracking_types.h"

namespace dt {

// ONNX Runtime based YOLO detector (generic; accepts detection ONNX exported by
// common toolchains). Auto-detects [1, C, N] or [1, N, C] output layouts, runs
// letterbox preprocessing, class filtering and NMS.
class YoloDetector {
public:
    struct Options {
        float conf_thresh = 0.25f;
        float iou_thresh = 0.45f;
        int target_class = 0;       // keep only this class (-1 = no filtering)
        bool filter_class = true;
        int imgsz = 640;
    };

    YoloDetector();
    ~YoloDetector() = default;

    bool load(const std::string& onnx_path);
    bool ok() const { return ok_; }

    // Run inference on one frame; returns detections (xyxy, original image coordinates)
    std::vector<Detection> detect(const cv::Mat& frame);

    Options opts;

private:
    // Decode the output tensor (auto-detect [1,C,N] or [1,N,C] layout), invert the
    // letterbox transform back to original coordinates, and filter by class.
    std::vector<Detection> decode(const float* data, int64_t dim1, int64_t dim2,
                                  float gain, float pad_x, float pad_y);
    // Per-class standard IoU non-maximum suppression.
    std::vector<Detection> nms(const std::vector<Detection>& dets);
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    Ort::Env env_;
    // Ort::Session holds a const Env& reference member and cannot be move-assigned,
    // so it is held through a unique_ptr.
    std::unique_ptr<Ort::Session> session_;
    bool ok_ = false;
    std::string input_name_;
    std::string output_name_;
    int imgsz_ = 640;
    int num_classes_ = 0;
};

} // namespace dt
