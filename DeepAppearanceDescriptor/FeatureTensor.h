#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "model.h"
#include "dataType.h"
typedef unsigned char uint8;

class FeatureTensor
{
public:
	static FeatureTensor* getInstance();
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);

private:
	FeatureTensor();
	FeatureTensor(const FeatureTensor&);
	FeatureTensor& operator = (const FeatureTensor&);
	static FeatureTensor* instance;
	bool init();
	~FeatureTensor();

	void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

	int feature_dim;
	tensorflow::Session* session;
	std::vector<tensorflow::Tensor> output_tensors;
	std::vector<tensorflow::string> outnames;
	tensorflow::string input_layer;
public:
	void test();
};
