#include "kalman_filter.h"

namespace dt {

KalmanFilterXYAH::KalmanFilterXYAH()
    : std_weight_position_(1.0 / 20.0), std_weight_velocity_(1.0 / 160.0) {
    const int ndim = 4;
    motion_mat_ = cv::Mat::eye(2 * ndim, 2 * ndim, CV_64F);
    for (int i = 0; i < ndim; ++i) motion_mat_.at<double>(i, ndim + i) = 1.0;
    update_mat_ = cv::Mat::eye(ndim, 2 * ndim, CV_64F);
}

void KalmanFilterXYAH::initiate(const cv::Mat& z, cv::Mat& mean, cv::Mat& cov) const {
    // z: 4x1 double (x,y,a,h)
    mean = cv::Mat::zeros(8, 1, CV_64F);
    z.copyTo(mean.rowRange(0, 4));
    double h = z.at<double>(3);   // z layout is (x, y, a, h): height is index 3
    double swp = std_weight_position_, swv = std_weight_velocity_;
    double std[8] = {
        2 * swp * h, 2 * swp * h, 1e-2, 2 * swp * h,
        10 * swv * h, 10 * swv * h, 1e-5, 10 * swv * h
    };
    cv::Mat s = cv::Mat(8, 1, CV_64F, std).clone();
    cov = cv::Mat::diag(s);
}

void KalmanFilterXYAH::predict(const cv::Mat& mean, const cv::Mat& cov, cv::Mat& mout, cv::Mat& cout) const {
    double h = mean.at<double>(3);
    double swp = std_weight_position_, swv = std_weight_velocity_;
    double sp[4] = {swp * h, swp * h, 1e-2, swp * h};
    double sv[4] = {swv * h, swv * h, 1e-5, swv * h};
    double sm_data[8] = {sp[0], sp[1], sp[2], sp[3], sv[0], sv[1], sv[2], sv[3]};
    cv::Mat sm = cv::Mat(1, 8, CV_64F, sm_data).clone();
    cv::Mat motion_cov = cv::Mat::diag(sm);  // 8x8

    mout = motion_mat_ * mean;
    cout = motion_mat_ * cov * motion_mat_.t() + motion_cov;
}

void KalmanFilterXYAH::project(const cv::Mat& mean, const cv::Mat& cov, cv::Mat& pmean, cv::Mat& pcov) const {
    double h = mean.at<double>(3);
    double swp = std_weight_position_;
    double sp[4] = {swp * h, swp * h, 1e-1, swp * h};
    cv::Mat sm = cv::Mat(1, 4, CV_64F, sp).clone();
    cv::Mat innov = cv::Mat::diag(sm);

    pmean = mean.rowRange(0, 4).clone();
    pcov = cov.rowRange(0, 4).colRange(0, 4).clone() + innov;
}

void KalmanFilterXYAH::update(const cv::Mat& mean, const cv::Mat& cov, const cv::Mat& z,
                              cv::Mat& mout, cv::Mat& cout) const {
    cv::Mat pmean, pcov;
    project(mean, cov, pmean, pcov);

    cv::Mat B = cov.colRange(0, 4).t();  // 4x8
    cv::Mat X;
    cv::solve(pcov, B, X, cv::DECOMP_LU);  // X: 4x8
    cv::Mat kalman_gain = X.t();           // 8x4

    cv::Mat innovation = z - pmean;        // 4x1
    cv::Mat correction = innovation.t() * kalman_gain.t();  // 1x8
    cv::Mat mean_row = mean.reshape(1, 1);                  // 1x8
    cv::Mat sum = mean_row + correction;
    mout = sum.reshape(1, 8).clone();   // 8x1

    cout = cov - kalman_gain * pcov * kalman_gain.t();      // 8x8
}

double KalmanFilterXYAH::mahalanobis2(const cv::Mat& mean, const cv::Mat& cov, const cv::Mat& z) const {
    cv::Mat pmean, pcov;
    project(mean, cov, pmean, pcov);
    cv::Mat d = z - pmean;                    // 4x1 innovation
    cv::Mat S_inv;
    if (!cv::invert(pcov, S_inv, cv::DECOMP_LU)) return 1e12;  // singular -> reject
    cv::Mat m2 = d.t() * S_inv * d;           // 1x1
    return m2.at<double>(0, 0);
}

} // namespace dt
