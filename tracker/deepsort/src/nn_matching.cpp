#include "nn_matching.h"


using namespace Eigen;

NearNeighborDisMetric::NearNeighborDisMetric(
    NearNeighborDisMetric::METRIC_TYPE metric,
    float matching_threshold, int budget)
{
  if(metric == euclidean)
    {
      _metric = &NearNeighborDisMetric::_nneuclidean_distance;
    } else if (metric == cosine)
    {
      _metric = &NearNeighborDisMetric::_nncosine_distance;
    }

  this->mating_threshold = matching_threshold;
  this->budget = budget;
  this->samples.clear();
}

DYNAMICM
NearNeighborDisMetric::distance(
    const FEATURESS &features,
    const std::vector<int>& targets)
{
  DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
  int idx = 0;
  for(int target:targets) {
      cost_matrix.row(idx) = (this->*_metric)(this->samples[target], features);
      idx++;
    }
  return cost_matrix;
}

void
NearNeighborDisMetric::partial_fit(
    std::vector<TRACKER_DATA> &tid_feats,
    std::vector<int> &active_targets)
{
  /*python code:
 * let feature(target_id) append to samples;
 * && delete not comfirmed target_id from samples.
 * update samples;
*/

  for(TRACKER_DATA& data:tid_feats) {
      int track_id = data.first;
      FEATURESS newFeatOne = data.second;

      if(samples.find(track_id) != samples.end()) {//append
          int oldSize = samples[track_id].rows();
          int addSize = newFeatOne.rows();
          int newSize = oldSize + addSize;

          if(newSize <= this->budget) {
              FEATURESS newSampleFeatures(newSize, k_feature_dim);
              newSampleFeatures.block(0,0, oldSize, k_feature_dim) = samples[track_id];
              newSampleFeatures.block(oldSize, 0, addSize, k_feature_dim) = newFeatOne;
              samples[track_id] = newSampleFeatures;
            } else {
              if(oldSize < this->budget) {//original space is not enough;
                  FEATURESS newSampleFeatures(this->budget, k_feature_dim);
                  if(addSize >= this->budget) {
                      newSampleFeatures = newFeatOne.block(0, 0, this->budget, k_feature_dim);
                    } else {
                      newSampleFeatures.block(0, 0, this->budget-addSize, k_feature_dim) =
                          samples[track_id].block(addSize-1, 0, this->budget-addSize, k_feature_dim).eval();
                      newSampleFeatures.block(this->budget-addSize, 0, addSize, k_feature_dim) = newFeatOne;
                    }
                  samples[track_id] = newSampleFeatures;
                } else {//original space is ok;
                  if(addSize >= this->budget) {
                      samples[track_id] = newFeatOne.block(0,0, this->budget, k_feature_dim);
                    } else {
                      samples[track_id].block(0, 0, this->budget-addSize, k_feature_dim) =
                          samples[track_id].block(addSize-1, 0, this->budget-addSize, k_feature_dim).eval();
                      samples[track_id].block(this->budget-addSize, 0, addSize, k_feature_dim) = newFeatOne;
                    }
                }
            }
        } else {//not exit, create new one;
          samples[track_id] = newFeatOne;
        }
    }//add features;

  //erase the samples which not in active_targets;
  for(std::map<int, FEATURESS>::iterator i = samples.begin(); i != samples.end();) {
      bool flag = false;
      for(int j:active_targets) if(j == i->first) { flag=true; break; }
      if(flag == false)  samples.erase(i++);
      else i++;
    }
}

Eigen::VectorXf
NearNeighborDisMetric::_nncosine_distance(
    const FEATURESS &x, const FEATURESS &y)
{
  MatrixXf distances = _cosine_distance(x,y);
  VectorXf res = distances.colwise().minCoeff().transpose();
  return res;
}

Eigen::VectorXf
NearNeighborDisMetric::_nneuclidean_distance(
    const FEATURESS &x, const FEATURESS &y)
{
  MatrixXf distances = _pdist(x,y);
  VectorXf res = distances.colwise().maxCoeff().transpose();
  res = res.array().max(VectorXf::Zero(res.rows()).array());
  return res;
}

Eigen::MatrixXf
NearNeighborDisMetric::_pdist(const FEATURESS &x, const FEATURESS &y)
{
  int len1 = x.rows(), len2 = y.rows();
  if(len1 == 0 || len2 == 0) {
      return Eigen::MatrixXf::Zero(len1, len2);
    }
  MatrixXf res = x * y.transpose()* -2;
  res = res.colwise() + x.rowwise().squaredNorm();
  res = res.rowwise() + y.rowwise().squaredNorm().transpose();
  res = res.array().max(MatrixXf::Zero(res.rows(), res.cols()).array());
  return res;
}

Eigen::MatrixXf
NearNeighborDisMetric::_cosine_distance(
    const FEATURESS & a,
    const FEATURESS& b, bool data_is_normalized) {
  if(data_is_normalized == true) {
      //undo:
      assert(false);
    }
  MatrixXf res = 1. - (a*b.transpose()).array();
  return res;
}
