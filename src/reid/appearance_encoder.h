#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace dt {

// ONNX Runtime based ReID appearance encoder: takes detection crops and outputs
// L2-normalized embedding vectors.
// Preprocessing matches common ReID models: BGR->RGB, /255.0, resize to the model
// input size, CHW float.
class AppearanceEncoder {
public:
    // model_path: .onnx path; device is currently fixed to CPU.
    explicit AppearanceEncoder(const std::string& model_path);
    ~AppearanceEncoder() = default;

    bool ok() const { return ok_; }

    // Extract features for several detection boxes (xyxy) in frame; returns an empty
    // vector for an invalid crop (caller treats it as missing).
    std::vector<std::vector<float>> operator()(const cv::Mat& frame,
                                               const std::vector<cv::Vec4f>& dets_xyxy);

private:
    std::vector<float> extract_one(const cv::Mat& crop);

    Ort::Env env_;
    // Ort::Session holds a const Env& reference member and cannot be move-assigned,
    // so it is held through a unique_ptr.
    std::unique_ptr<Ort::Session> session_;
    bool ok_ = false;
    std::string input_name_;
    std::string output_name_;
    int in_h_ = 224, in_w_ = 224;
};

} // namespace dt
