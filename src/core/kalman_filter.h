#pragma once
#include <opencv2/opencv.hpp>

namespace dt {

// 8-dimensional state Kalman filter (XYAH layout): (x, y, a, h, vx, vy, va, vh)
// Constant-velocity motion model with a linear observation model (XYAH Kalman design).
class KalmanFilterXYAH {
public:
    KalmanFilterXYAH();

    // Initialize mean(8x1) and cov(8x8) from a measurement z(4x1: x,y,a,h)
    void initiate(const cv::Mat& z, cv::Mat& mean, cv::Mat& cov) const;

    // Predict step: mean(8x1), cov(8x8) -> mean_out, cov_out
    void predict(const cv::Mat& mean, const cv::Mat& cov, cv::Mat& mean_out, cv::Mat& cov_out) const;

    // Project the state distribution into the 4-d observation space
    void project(const cv::Mat& mean, const cv::Mat& cov, cv::Mat& pmean, cv::Mat& pcov) const;

    // Update step: z(4x1) is the measurement (x,y,a,h)
    void update(const cv::Mat& mean, const cv::Mat& cov, const cv::Mat& z,
                cv::Mat& mean_out, cv::Mat& cov_out) const;

    // Squared Mahalanobis distance of measurement z(4x1: x,y,a,h) under the
    // projected (observation-space) distribution of (mean, cov). Used by
    // classic DeepSORT motion gating (chi2, 4 dof).
    double mahalanobis2(const cv::Mat& mean, const cv::Mat& cov, const cv::Mat& z) const;

private:
    cv::Mat motion_mat_;   // 8x8
    cv::Mat update_mat_;   // 4x8
    double std_weight_position_;
    double std_weight_velocity_;
};

} // namespace dt
