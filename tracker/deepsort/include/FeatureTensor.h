
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
#include "opencv2/opencv.hpp"
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

private:
    FeatureTensor();
    FeatureTensor(const FeatureTensor &);
    FeatureTensor &operator=(const FeatureTensor &);
    static FeatureTensor *instance;
    bool init();
    ~FeatureTensor();

    void preprocess(cv::Mat &imageBGR);
public:
    static constexpr const int width_ = 64;
    static constexpr const int height_ = 128;

    cv::dnn::Net net;
};
