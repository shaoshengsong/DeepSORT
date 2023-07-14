/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 04:32:26
*/

//#include "globalconfig.h"
#include "FeatureTensor.h"
#include <iostream>

FeatureTensor *FeatureTensor::instance = NULL;

FeatureTensor *FeatureTensor::getInstance()
{
    if (instance == NULL)
    {
        instance = new FeatureTensor();
    }
    return instance;
}

FeatureTensor::FeatureTensor()
{
    // prepare model:
    bool status = init();
    if (status == false)
    {
        std::cout << "init failed" << std::endl;
        exit(1);
    }
    else
    {
        std::cout << "init succeed" << std::endl;
    }
}

FeatureTensor::~FeatureTensor()
{
}

bool FeatureTensor::init()
{

    this->net = cv::dnn::readNetFromONNX(k_feature_model_path);
    std::cout << "FeatureTensor::init() " << std::endl;
    return true;
}

void FeatureTensor::preprocess(cv::Mat &imageBGR)
{

    // pre-processing the Image
    //  step 1: Read an image in HWC BGR UINT8 format.
    //  cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

    // step 2: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
                cv::Size(width_, height_),
                cv::InterpolationFlags::INTER_CUBIC);

    // cv::resize(imageBGR, resizedImageBGR,
    //            cv::Size(64, 128));

    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // step 5: Split the RGB channels from the image.
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    // step 6: Normalize each channel.
    //  Normalization per channel
    //  Normalization parameters obtained from your custom model

    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    // step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    this->net.setInput(preprocessedImage);
}

bool FeatureTensor::getRectsFeature(const cv::Mat &img, DETECTIONS &d)
{

    for (DETECTION_ROW &dbox : d)
    {
        cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                               int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
        rc.width = rc.height * 0.5;
        rc.x = (rc.x >= 0 ? rc.x : 0);
        rc.y = (rc.y >= 0 ? rc.y : 0);
        rc.width = (rc.x + rc.width <= img.cols ? rc.width : (img.cols - rc.x));
        rc.height = (rc.y + rc.height <= img.rows ? rc.height : (img.rows - rc.y));

        cv::Mat mattmp = img(rc).clone();
        preprocess(mattmp);

        cv::Mat preds = this->net.forward("output");
        auto *ptr = preds.ptr<float>(0);

        for (int i = 0; i < preds.cols; i++)
        {
            dbox.feature[i] = ptr[i];
        }
    }

    return true;
}
