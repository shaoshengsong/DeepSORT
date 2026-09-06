#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

namespace dt {

// ---- Box format conversion (all based on floating-point xyxy = [x1,y1,x2,y2]) ----
inline cv::Vec4f tlwh_to_xyxy(const cv::Vec4f& tlwh) {
    return cv::Vec4f(tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]);
}
inline cv::Vec4f xyxy_to_tlwh(const cv::Vec4f& xyxy) {
    return cv::Vec4f(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]);
}
// tlwh -> xyah (center_x, center_y, aspect=a=w/h, height)
inline cv::Vec4f tlwh_to_xyah(const cv::Vec4f& tlwh) {
    float w = tlwh[2], h = std::max(tlwh[3], 1e-6f);
    return cv::Vec4f(tlwh[0] + w / 2.0f, tlwh[1] + h / 2.0f, w / h, h);
}
inline cv::Vec4f xyah_to_tlwh(const cv::Vec4f& xyah) {
    float a = xyah[2], h = xyah[3];
    float w = a * h;
    return cv::Vec4f(xyah[0] - w / 2.0f, xyah[1] - h / 2.0f, w, h);
}
inline cv::Vec4f xywh_center_to_tlwh(const cv::Vec4f& xywh) {
    // xywh = (cx, cy, w, h)
    return cv::Vec4f(xywh[0] - xywh[2] / 2.0f, xywh[1] - xywh[3] / 2.0f, xywh[2], xywh[3]);
}
inline cv::Point2f xyxy_center(const cv::Vec4f& xyxy) {
    return cv::Point2f((xyxy[0] + xyxy[2]) / 2.0f, (xyxy[1] + xyxy[3]) / 2.0f);
}

// Axis-aligned IoU of two boxes
inline float iou_xyxy(const cv::Vec4f& a, const cv::Vec4f& b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float area_b = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    float uni = area_a + area_b - inter;
    return uni > 1e-6f ? inter / uni : 0.0f;
}

// IoU distance matrix cost = 1 - iou, shape (a.size() x b.size())
inline void iou_distance_matrix(const std::vector<cv::Vec4f>& a,
                               const std::vector<cv::Vec4f>& b,
                               std::vector<std::vector<float>>& cost) {
    cost.assign(a.size(), std::vector<float>(b.size(), 0.0f));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            cost[i][j] = 1.0f - iou_xyxy(a[i], b[j]);
}

} // namespace dt
