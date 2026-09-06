#include "yolo_detector.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <fstream>
#include <cstdint>

namespace dt {

YoloDetector::YoloDetector() {}

bool YoloDetector::load(const std::string& onnx_path) {
    ok_ = false;
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo_det");
        // On Windows ORTCHAR_T is wchar_t, so path loading needs wide chars.
        // Load from memory instead for consistent cross-platform behavior.
        std::ifstream fmodel(onnx_path, std::ios::binary);
        if (!fmodel) throw std::runtime_error("Failed to open YOLO model: " + onnx_path);
        std::vector<uint8_t> blob((std::istreambuf_iterator<char>(fmodel)),
                                  std::istreambuf_iterator<char>());
        if (blob.empty()) throw std::runtime_error("YOLO model is empty: " + onnx_path);
        session_ = std::make_unique<Ort::Session>(env_, blob.data(), blob.size(), opts);

        // ONNX Runtime 1.28: use the Allocated variants to fetch input/output
        // names (they return unique_ptr<char>).
        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = std::string(session_->GetInputNameAllocated(0, allocator).get());
        output_name_ = std::string(session_->GetOutputNameAllocated(0, allocator).get());
        ok_ = true;
    } catch (const std::exception& e) {
        (void)e;
        ok_ = false;
    }
    return ok_;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    std::vector<Detection> empty;
    if (!ok_ || frame.empty()) return empty;

    const int in_w = opts.imgsz;
    const int in_h = opts.imgsz;

    // ---- letterbox preprocessing ----
    const float r = std::min((float)in_w / (float)frame.cols, (float)in_h / (float)frame.rows);
    const int new_w = (int)std::round(frame.cols * r);
    const int new_h = (int)std::round(frame.rows * r);
    const int pad_x = (in_w - new_w) / 2;
    const int pad_y = (in_h - new_h) / 2;

    cv::Mat tmp;
    cv::resize(frame, tmp, cv::Size(new_w, new_h));
    cv::Mat resized(in_h, in_w, CV_8UC3, cv::Scalar(114, 114, 114));
    tmp.copyTo(resized(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // Convert to CHW float, BGR->RGB, /255
    cv::Mat fimg;
    resized.convertTo(fimg, CV_32F, 1.0 / 255.0);
    cv::cvtColor(fimg, fimg, cv::COLOR_BGR2RGB);

    const int wh = in_w * in_h;
    std::vector<float> input_data(static_cast<size_t>(wh) * 3);
    const float* src = reinterpret_cast<const float*>(fimg.data);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < wh; ++i)
            input_data[(size_t)c * wh + i] = src[(size_t)i * 3 + c];

    // ---- inference ----
    std::array<int64_t, 4> input_shape{1, 3, in_h, in_w};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    const char* in_names[1] = {input_name_.c_str()};
    const char* out_names[1] = {output_name_.c_str()};

    Ort::RunOptions run_opts;
    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(run_opts, in_names, &input_tensor, 1, out_names, 1);
    } catch (const std::exception&) {
        return empty;
    }
    if (outputs.empty()) return empty;

    const float* data = outputs[0].GetTensorMutableData<float>();
    auto shape_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto shape = shape_info.GetShape();
    if (shape.size() != 3) return empty;
    // shape = [1, dim1, dim2]
    int64_t dim1 = shape[1];
    int64_t dim2 = shape[2];

    auto decoded = decode(data, dim1, dim2, r, (float)pad_x, (float)pad_y);
    // Layout check must mirror decode(): [1, 300, 6] (or generally N x 6 with
    // N <= dim1) is the end-to-end NMS-fused output - the model has already
    // deduplicated, so running NMS again would wrongly suppress valid
    // closely-overlapping objects (missed detections, lower mAP).
    bool cn_layout = (dim2 >= dim1);
    const int C = cn_layout ? (int)dim1 : (int)dim2;
    const bool end2end = (C == 6);
    if (end2end) return decoded;   // already NMS'd inside the model
    return nms(decoded);           // raw detection head -> NMS here
}

std::vector<Detection> YoloDetector::decode(const float* data, int64_t dim1, int64_t dim2,
                                            float gain, float pad_x, float pad_y) {
    // Determine layout: detection-export ONNX is usually [1, C, N]
    // (C=4+nc is large, N is large); an end-to-end (NMS-fused) model is
    // [1, N, C] with C=6 and each row = [x1,y1,x2,y2,score,class].
    bool cn_layout = (dim2 >= dim1);   // [C, N] otherwise [N, C]
    const int C = cn_layout ? (int)dim1 : (int)dim2;
    const int N = cn_layout ? (int)dim2 : (int)dim1;
    const bool end2end = (C == 6);     // NMS-fused output
    const int nc = end2end ? 0 : (C - 4);
    if (!end2end && nc <= 0) return {};

    // Value accessor: coord is the channel (0..C-1), n is the candidate index.
    auto get = [&](int coord, int n) -> float {
        if (cn_layout) return data[(size_t)coord * N + n];
        return data[(size_t)n * C + coord];
    };

    std::vector<Detection> out;
    out.reserve(N);
    for (int n = 0; n < N; ++n) {
        float a = get(0, n), b = get(1, n), c = get(2, n), d = get(3, n);
        float score; int cls;
        if (end2end) {
            score = get(4, n);
            cls = (int)std::round(get(5, n));
        } else {
            int best = -1; float best_s = -1.f;
            for (int k = 0; k < nc; ++k) {
                float s = get(4 + k, n);
                if (s > best_s) { best_s = s; best = k; }
            }
            if (best < 0) continue;
            score = best_s; cls = best;
        }
        if (opts.filter_class && cls != opts.target_class) continue;
        if (score < opts.conf_thresh) continue;

        // Reverse letterbox: network coords (640 space) -> original image coords.
        float x1, y1, x2, y2;
        if (end2end) {
            // a,b,c,d are already corners (x1,y1,x2,y2)
            x1 = (a - pad_x) / gain;
            y1 = (b - pad_y) / gain;
            x2 = (c - pad_x) / gain;
            y2 = (d - pad_y) / gain;
        } else {
            // a,b,c,d are center cx,cy,w,h
            float cx = (a - pad_x) / gain;
            float cy = (b - pad_y) / gain;
            float bw = c / gain, bh = d / gain;
            x1 = cx - bw * 0.5f; y1 = cy - bh * 0.5f;
            x2 = cx + bw * 0.5f; y2 = cy + bh * 0.5f;
        }
        if (x2 <= x1 || y2 <= y1) continue;

        Detection det;
        det.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
        det.conf = score;
        det.cls = cls;
        det.idx = (int)out.size();
        out.push_back(det);
    }
    return out;
}

float YoloDetector::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float ix1 = std::max(a.x, b.x);
    float iy1 = std::max(a.y, b.y);
    float ix2 = std::min(a.x + a.width, b.x + b.width);
    float iy2 = std::min(a.y + a.height, b.y + b.height);
    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float u = area_a + area_b - inter;
    if (u <= 0.f) return 0.f;
    return inter / u;
}

std::vector<Detection> YoloDetector::nms(const std::vector<Detection>& dets) {
    std::map<int, std::vector<int>> by_cls;
    for (int i = 0; i < (int)dets.size(); ++i)
        by_cls[dets[i].cls].push_back(i);

    std::vector<char> keep(dets.size(), 0);
    for (auto& kv : by_cls) {
        std::vector<int>& idxs = kv.second;
        std::sort(idxs.begin(), idxs.end(),
                  [&](int a, int b) { return dets[a].conf > dets[b].conf; });
        for (size_t i = 0; i < idxs.size(); ++i) {
            if (keep[idxs[i]]) continue;
            keep[idxs[i]] = 1;
            for (size_t j = i + 1; j < idxs.size(); ++j) {
                if (keep[idxs[j]]) continue;
                if (iou(dets[idxs[i]].box, dets[idxs[j]].box) > opts.iou_thresh)
                    keep[idxs[j]] = 1;
            }
        }
    }

    std::vector<Detection> out;
    for (int i = 0; i < (int)dets.size(); ++i)
        if (keep[i]) out.push_back(dets[i]);
    return out;
}

} // namespace dt
