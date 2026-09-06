#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace dt {

// Global Motion Compensation: estimates the 2x3 affine transform between
// consecutive frames to correct for camera motion.
// Supports sparseOptFlow / orb / ecc / none (none returns the identity matrix).
class GlobalMotionCompensation {
public:
    explicit GlobalMotionCompensation(const std::string& method = "sparseOptFlow", int downscale = 2);
    ~GlobalMotionCompensation() = default;

    // Estimate the warp (2x3, CV_32F) from the previous frame to the current frame.
    // detections are xyxy boxes used to mask out occluded regions.
    cv::Mat apply(const cv::Mat& raw_frame, const std::vector<cv::Vec4f>& detections);

    void reset_params();

private:
    cv::Mat apply_ecc(const cv::Mat& frame);
    cv::Mat apply_features(const cv::Mat& frame, const std::vector<cv::Vec4f>& detections);
    cv::Mat apply_sparseoptflow(const cv::Mat& frame);

    std::string method_;
    int downscale_;
    cv::Mat prev_frame_;
    std::vector<cv::KeyPoint> prev_kpts_;
    cv::Mat prev_desc_;
    bool initialized_ = false;

    // Used by the feature-based (orb) method
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::Feature2D> extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

} // namespace dt
