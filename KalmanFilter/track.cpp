#include "track.h"

Track::Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int n_init, int max_age, const FEATURE& feature)
{
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = track_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = FEATURESS(1, 128);
    features.row(0) = feature;//features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = max_age;
}

void Track::predit(KalmanFilter *kf)
{
    /*Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        */

    kf->predict(this->mean, this->covariance);
    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter * const kf, const DETECTION_ROW& detection)
{
    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;

    featuresAppendOne(detection.feature);
    //    this->features.row(features.rows()) = detection.feature;
    this->hits += 1;
    this->time_since_update = 0;
    if(this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
}

void Track::mark_missed()
{
    if(this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if(this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::is_confirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative()
{
    return this->state == TrackState::Tentative;
}

DETECTBOX Track::to_tlwh()
{
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2)/2);
    return ret;
}

void Track::featuresAppendOne(const FEATURE &f)
{
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size+1, 128);
    newfeatures.block(0, 0, size, 128) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}
