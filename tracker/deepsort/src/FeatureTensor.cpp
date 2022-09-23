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

    Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    inputDims_ = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims_ << std::endl; // [-1, 3, 128, 64]
    inputDims_[0] = 1;
    std::cout << "FeatureTensor::init() " << std::endl;


    return true;
}

void FeatureTensor::preprocess(cv::Mat &imageBGR, std::vector<float> &inputTensorValues, size_t &inputTensorSize)
{

    // pre-processing the Image
    //  step 1: Read an image in HWC BGR UINT8 format.
    //  cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

    // step 2: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
       cv::resize(imageBGR, resizedImageBGR,
                  cv::Size(inputDims_.at(3), inputDims_.at(2)),
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
    inputTensorSize = vectorProduct(inputDims_);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    std::cout << "inputTensorSize:" << inputTensorValues.size() << std::endl;
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

        std::vector<float> inputTensorValues;
        size_t inputTensorSize;
        preprocess(mattmp, inputTensorValues, inputTensorSize);

        const char *input_names[] = {"input"};   //输入节点名
        const char *output_names[] = {"output"}; //输出节点名

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, inputTensorValues.data(), inputTensorSize, inputDims_.data(),
            inputDims_.size()));


        session_.Run(Ort::RunOptions{nullptr}, input_names, inputTensors.data(), 1, output_names, &output_tensor_, 1);
     
        Ort::TensorTypeAndShapeInfo shape_info = output_tensor_.GetTensorTypeAndShapeInfo();


        size_t dim_count = shape_info.GetDimensionsCount();
        std::cout << "dim_count:" << dim_count << std::endl;

  
        int64_t dims[2];
        shape_info.GetDimensions(dims, sizeof(dims) / sizeof(dims[0]));
        std::cout << "output shape:" << dims[0] << "," << dims[1] << std::endl;


        float *f = output_tensor_.GetTensorMutableData<float>();
        for (int i = 0; i < dims[1]; i++) //sisyphus
        {
            dbox.feature[i] = f[i];
        }
    }

    return true;
}

void FeatureTensor::tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf)
{
    int pos = 0;
    for (const cv::Mat &img : imgs)
    {
        int Lenth = img.rows * img.cols * 3;
        int nr = img.rows;
        int nc = img.cols;
        if (img.isContinuous())
        {
            nr = 1;
            nc = Lenth;
        }
        for (int i = 0; i < nr; i++)
        {
            const uchar *inData = img.ptr<uchar>(i);
            for (int j = 0; j < nc; j++)
            {
                buf[pos] = *inData++;
                pos++;
            }
        } // end for
    }     // end imgs;
}
void FeatureTensor::test()
{
    return;
}
