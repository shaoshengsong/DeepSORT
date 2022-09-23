#ifndef NN_MATCHING_H
#define NN_MATCHING_H

#include "dataType.h"

#include <map>

//A tool to calculate distance;
class NearNeighborDisMetric{
public:
    enum METRIC_TYPE{euclidean=1, cosine};
    NearNeighborDisMetric(METRIC_TYPE metric,
            float matching_threshold,
            int budget);
    DYNAMICM distance(const FEATURESS& features, const std::vector<int> &targets);
    //    void partial_fit(FEATURESS& features, std::vector<int> targets, std::vector<int> active_targets);
    void partial_fit(std::vector<TRACKER_DATA>& tid_feats, std::vector<int>& active_targets);
    float mating_threshold;

private:
    typedef Eigen::VectorXf (NearNeighborDisMetric::*PTRFUN)(const FEATURESS&, const FEATURESS&);
    Eigen::VectorXf _nncosine_distance(const FEATURESS& x, const FEATURESS& y);
    Eigen::VectorXf _nneuclidean_distance(const FEATURESS& x, const FEATURESS& y);

    Eigen::MatrixXf _pdist(const FEATURESS& x, const FEATURESS& y);
    Eigen::MatrixXf _cosine_distance(const FEATURESS & a, const FEATURESS& b, bool data_is_normalized = false);
private:
    PTRFUN _metric;
    int budget;
    std::map<int, FEATURESS > samples;
};

#endif // NN_MATCHING_H
