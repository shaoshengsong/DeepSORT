#include "global_motion_compensation.h"
#include <opencv2/features2d.hpp>
#include <opencv2/geometry/3d.hpp>

namespace dt {

GlobalMotionCompensation::GlobalMotionCompensation(const std::string& method, int downscale)
    : method_(method), downscale_(downscale > 0 ? downscale : 1) {
    if (method_ == "orb") {
        detector_ = cv::FastFeatureDetector::create(20);
        extractor_ = cv::ORB::create();
        matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
    } else if (method_ == "sift") {
        // SIFT ships with the main OpenCV 5 module; fall back to sparseOptFlow
        // if it is unavailable.
        try {
            extractor_ = cv::SIFT::create(0, 3, 0.02, 20);
            detector_ = extractor_;
            matcher_ = cv::BFMatcher::create(cv::NORM_L2);
        } catch (...) {
            method_ = "sparseOptFlow";
        }
    }
    if (method_ == "sift" || method_ == "orb") {
        // Use the extractor as the detector as well.
        if (!detector_) detector_ = extractor_;
    }
}

void GlobalMotionCompensation::reset_params() {
    prev_frame_.release();
    prev_kpts_.clear();
    prev_desc_.release();
    initialized_ = false;
}

cv::Mat GlobalMotionCompensation::apply(const cv::Mat& raw_frame, const std::vector<cv::Vec4f>& detections) {
    if (method_ == "none" || method_.empty()) return cv::Mat::eye(2, 3, CV_32F);
    if (method_ == "orb" || method_ == "sift") return apply_features(raw_frame, detections);
    if (method_ == "ecc") return apply_ecc(raw_frame);
    return apply_sparseoptflow(raw_frame);
}

cv::Mat GlobalMotionCompensation::apply_ecc(const cv::Mat& frame) {
    int h = frame.rows, w = frame.cols;
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame.clone();
    cv::Mat H = cv::Mat::eye(2, 3, CV_32F);

    if (downscale_ > 1) {
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.5);
        cv::resize(gray, gray, cv::Size(w / downscale_, h / downscale_));
    }
    if (!initialized_) {
        prev_frame_ = gray.clone();
        initialized_ = true;
        return H;
    }
    try {
        float warp_data[6] = {1, 0, 0, 0, 1, 0};
        cv::Mat warp = cv::Mat(2, 3, CV_32F, warp_data).clone();
        cv::TermCriteria crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 5000, 1e-6);
        cv::findTransformECC(prev_frame_, gray, warp, cv::MOTION_EUCLIDEAN, crit);
        H = warp;
        H.at<float>(0, 2) *= static_cast<float>(w) / static_cast<float>(gray.cols);
        H.at<float>(1, 2) *= static_cast<float>(h) / static_cast<float>(gray.rows);
    } catch (...) {
        // On failure fall back to the identity matrix.
    }
    prev_frame_ = gray.clone();
    return H;
}

cv::Mat GlobalMotionCompensation::apply_features(const cv::Mat& frame, const std::vector<cv::Vec4f>& detections) {
    int h = frame.rows, w = frame.cols;
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame.clone();
    cv::Mat H = cv::Mat::eye(2, 3, CV_32F);
    cv::Mat small = gray;
    if (downscale_ > 1) cv::resize(gray, small, cv::Size(w / downscale_, h / downscale_));

    cv::Mat mask = cv::Mat::zeros(small.size(), CV_8U);
    mask(cv::Rect(0, int(0.02 * small.rows), small.cols, int(0.96 * small.rows))).setTo(255);
    mask(cv::Rect(int(0.02 * small.cols), 0, int(0.96 * small.cols), small.rows)).setTo(255);
    for (const auto& d : detections) {
        int x1 = static_cast<int>(d[0] / downscale_), y1 = static_cast<int>(d[1] / downscale_);
        int x2 = static_cast<int>(d[2] / downscale_), y2 = static_cast<int>(d[3] / downscale_);
        cv::Rect r(x1, y1, x2 - x1, y2 - y1);
        if (r.x >= 0 && r.y >= 0 && r.br().x <= mask.cols && r.br().y <= mask.rows)
            mask(r) = 0;
    }

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    detector_->detect(small, kpts, mask);
    if (kpts.empty()) {
        prev_frame_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = cv::Mat();
        return H;
    }
    extractor_->compute(small, kpts, desc);

    if (!initialized_) {
        prev_frame_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = desc.clone();
        initialized_ = true;
        return H;
    }

    if (prev_desc_.empty() || desc.empty()) {
        prev_frame_ = small.clone();
        prev_kpts_ = kpts;
        prev_desc_ = desc.clone();
        return H;
    }

    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(prev_desc_, desc, knn, 2);
    std::vector<cv::Point2f> prevPts, currPts;
    for (const auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.9 * m[1].distance) {
            cv::Point2f p = prev_kpts_[m[0].queryIdx].pt;
            cv::Point2f c = kpts[m[0].trainIdx].pt;
            cv::Point2f d = p - c;
            if (std::abs(d.x) < 0.25 * small.cols && std::abs(d.y) < 0.25 * small.rows) {
                prevPts.push_back(p);
                currPts.push_back(c);
            }
        }
    }
    if (prevPts.size() > 4) {
        cv::Mat mask_ransac;
        cv::Mat Haff = cv::estimateAffinePartial2D(prevPts, currPts, mask_ransac);
        if (!Haff.empty()) {
            H = Haff;
            if (downscale_ > 1) {
                H.at<float>(0, 2) *= static_cast<float>(downscale_);
                H.at<float>(1, 2) *= static_cast<float>(downscale_);
            }
        }
    }
    prev_frame_ = small.clone();
    prev_kpts_ = kpts;
    prev_desc_ = desc.clone();
    return H;
}

cv::Mat GlobalMotionCompensation::apply_sparseoptflow(const cv::Mat& frame) {
    int h = frame.rows, w = frame.cols;
    cv::Mat gray;
    if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else gray = frame.clone();
    cv::Mat H = cv::Mat::eye(2, 3, CV_32F);
    cv::Mat small = gray;
    if (downscale_ > 1) cv::resize(gray, small, cv::Size(w / downscale_, h / downscale_));

    cv::Mat kp;
    cv::goodFeaturesToTrack(small, kp, 400, 0.01, 0.0, cv::noArray(), 3, false, 0.04);
    if (!initialized_ || kp.empty()) {
        prev_frame_ = small.clone();
        prev_kpts_.clear();
        if (!kp.empty()) {
            cv::Mat_<cv::Point2f> km(kp);
            for (const auto& p : km) prev_kpts_.push_back(cv::KeyPoint(p, 1.f));
        }
        initialized_ = true;
        return H;
    }

    std::vector<cv::Point2f> prevPts_in;
    prevPts_in.reserve(prev_kpts_.size());
    for (auto& k : prev_kpts_) prevPts_in.push_back(k.pt);
    cv::Mat matched, status, err;
    cv::calcOpticalFlowPyrLK(prev_frame_, small, prevPts_in, matched, status, err);
    std::vector<cv::Point2f> prevPts, currPts;
    for (size_t i = 0; i < prevPts_in.size(); ++i) {
        if ((int)i < status.rows && status.at<uchar>(static_cast<int>(i)) > 0) {
            prevPts.push_back(prevPts_in[i]);
            currPts.push_back(matched.at<cv::Point2f>(static_cast<int>(i)));
        }
    }
    if (prevPts.size() > 4) {
        cv::Mat mask_r;
        cv::Mat Haff = cv::estimateAffinePartial2D(prevPts, currPts, mask_r);
        if (!Haff.empty()) {
            H = Haff;
            if (downscale_ > 1) {
                H.at<float>(0, 2) *= static_cast<float>(downscale_);
                H.at<float>(1, 2) *= static_cast<float>(downscale_);
            }
        }
    }
    prev_frame_ = small.clone();
    prev_kpts_.clear();
    if (!kp.empty()) {
        cv::Mat_<cv::Point2f> km(kp);
        for (const auto& p : km) prev_kpts_.push_back(cv::KeyPoint(p, 1.f));
    }
    return H;
}

} // namespace dt
