#include "appearance_encoder.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <cstdint>

namespace dt {

AppearanceEncoder::AppearanceEncoder(const std::string& model_path) : session_(nullptr) {
    try {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "reid");
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(1);
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        // On Windows ORTCHAR_T is wchar_t, so path loading needs wide chars.
        // Load from memory instead for consistent cross-platform behavior.
        std::ifstream fmodel(model_path, std::ios::binary);
        if (!fmodel) throw std::runtime_error("Failed to open ReID model: " + model_path);
        std::vector<uint8_t> blob((std::istreambuf_iterator<char>(fmodel)),
                                  std::istreambuf_iterator<char>());
        if (blob.empty()) throw std::runtime_error("ReID model is empty: " + model_path);
        session_ = std::make_unique<Ort::Session>(env_, blob.data(), blob.size(), so);

        // ONNX Runtime 1.28: use the Allocated variants to fetch input/output
        // names (they return unique_ptr<char>).
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = std::string(session_->GetInputNameAllocated(0, allocator).get());
        output_name_ = std::string(session_->GetOutputNameAllocated(0, allocator).get());

        auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4) {
            if (shape[2] > 0) in_h_ = static_cast<int>(shape[2]);
            if (shape[3] > 0) in_w_ = static_cast<int>(shape[3]);
        }
        ok_ = true;
    } catch (const std::exception& e) {
        (void)e;
        ok_ = false;
    }
}

std::vector<float> AppearanceEncoder::extract_one(const cv::Mat& crop) {
    cv::Mat rgb, f;
    if (crop.channels() == 3) cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
    else rgb = crop.clone();
    cv::resize(rgb, rgb, cv::Size(in_w_, in_h_), cv::INTER_LINEAR);
    rgb.convertTo(f, CV_32F, 1.0 / 255.0);

    std::vector<float> buf(static_cast<size_t>(3 * in_h_ * in_w_));
    // HWC -> CHW
    const int H = in_h_, W = in_w_;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                buf[c * H * W + i * W + j] = f.at<float>(i, j * 3 + c);

    std::vector<int64_t> in_shape = {1, 3, H, W};
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(mi, buf.data(), buf.size(),
                                                        in_shape.data(), in_shape.size());

    std::vector<const char*> inames = {input_name_.c_str()};
    std::vector<const char*> onames = {output_name_.c_str()};
    Ort::RunOptions runopt;
    std::vector<Ort::Value> out;
    try {
        out = session_->Run(runopt, inames.data(), &tensor, 1, onames.data(), 1);
    } catch (const std::exception&) {
        return {};  // Inference failed (e.g. input size mismatch) -> empty feature.
    }
    float* ptr = out[0].GetTensorMutableData<float>();
    size_t n = static_cast<size_t>(out[0].GetTensorTypeAndShapeInfo().GetElementCount());

    std::vector<float> feat(ptr, ptr + n);
    // L2 normalize
    double s = 0.0;
    for (float v : feat) s += static_cast<double>(v) * v;
    s = std::sqrt(s);
    if (s > 1e-12) for (float& v : feat) v = static_cast<float>(v / s);
    return feat;
}

std::vector<std::vector<float>> AppearanceEncoder::operator()(const cv::Mat& frame,
                                                        const std::vector<cv::Vec4f>& dets_xyxy) {
    std::vector<std::vector<float>> feats;
    feats.reserve(dets_xyxy.size());
    if (!ok_) {
        for (size_t i = 0; i < dets_xyxy.size(); ++i) feats.emplace_back();
        return feats;
    }
    for (const auto& d : dets_xyxy) {
        int x1 = static_cast<int>(std::max(0.f, d[0]));
        int y1 = static_cast<int>(std::max(0.f, d[1]));
        int x2 = static_cast<int>(std::min(static_cast<float>(frame.cols), d[2]));
        int y2 = static_cast<int>(std::min(static_cast<float>(frame.rows), d[3]));
        cv::Rect r(x1, y1, x2 - x1, y2 - y1);
        if (r.width <= 0 || r.height <= 0) {
            feats.emplace_back();  // Invalid crop
            continue;
        }
        feats.push_back(extract_one(frame(r)));
    }
    return feats;
}

} // namespace dt
