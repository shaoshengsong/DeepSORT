
/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 02:39:47
*/
#include "model.h"
#include "dataType.h"
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept> 
#include <onnxruntime_cxx_api.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
typedef unsigned char uint8;

template <typename T>
T vectorProduct(const std::vector<T> &v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
class FeatureTensor
{
public:
    static FeatureTensor *getInstance();
    bool getRectsFeature(const cv::Mat &img, DETECTIONS &d);
    void preprocess(cv::Mat &imageBGR, std::vector<float> &inputTensorValues, size_t &inputTensorSize);

private:
    FeatureTensor();
    FeatureTensor(const FeatureTensor &);
    FeatureTensor &operator=(const FeatureTensor &);
    static FeatureTensor *instance;
    bool init();
    ~FeatureTensor();

    void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

public:
    void test();

    static constexpr const int width_ = 64;
    static constexpr const int height_ = 128;

    std::array<float, width_ * height_> input_image_{};

    std::array<float, k_feature_dim> results_{};

    Ort::Env env;
    Ort::Session session_{env, k_feature_model_path.c_str(), Ort::SessionOptions{nullptr}};

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 3, width_, height_};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, k_feature_dim};

    std::vector<int64_t> inputDims_;
};
