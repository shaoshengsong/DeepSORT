#include "tracked_object.h"
#include <algorithm>

namespace dt {

int TrackedObjectBase::_count = 0;
KalmanFilterXYAH SortTrack::shared_kalman;

// ---------------- SortTrack ----------------
SortTrack::SortTrack(const cv::Vec4f& xywh, float score, int cls, int idx, float angle)
    : cls(cls), idx(idx), angle(angle) {
    this->score = score;  // score lives in base TrackedObjectBase, so it cannot appear in the member-init list
    // xywh = (cx, cy, w, h) -> tlwh
    _tlwh = cv::Vec4f(xywh[0] - xywh[2] / 2.0f, xywh[1] - xywh[3] / 2.0f, xywh[2], xywh[3]);
}

cv::Vec4f SortTrack::convert_coords(const cv::Vec4f& tlwh) const {
    return tlwh_to_xyah(tlwh);
}

cv::Vec4f SortTrack::tlwh_to_xyah(const cv::Vec4f& tlwh) {
    return dt::tlwh_to_xyah(tlwh);
}

cv::Vec4f SortTrack::tlwh() const {
    if (!has_kalman) return _tlwh;
    cv::Vec4f ret;
    for (int i = 0; i < 4; ++i) ret[i] = static_cast<float>(mean.at<double>(i));
    ret[2] *= ret[3];                 // a*h -> w
    ret[0] -= ret[2] / 2.0f;          // cx - w/2
    ret[1] -= ret[3] / 2.0f;          // cy - h/2
    return ret;
}

cv::Vec4f SortTrack::xyxy() const {
    cv::Vec4f t = tlwh();
    return cv::Vec4f(t[0], t[1], t[0] + t[2], t[1] + t[3]);
}

cv::Vec4f SortTrack::xywh() const {
    cv::Vec4f t = tlwh();
    return cv::Vec4f(t[0] + t[2] / 2.0f, t[1] + t[3] / 2.0f, t[2], t[3]);
}

std::vector<float> SortTrack::result() const {
    cv::Vec4f b = xyxy();
    return {b[0], b[1], b[2], b[3], static_cast<float>(track_id), score,
            static_cast<float>(cls), static_cast<float>(idx)};
}

void SortTrack::predict() {
    if (!kf) return;
    cv::Mat m = mean.clone();
    if (state != TrackState::Tracked) m.at<double>(7) = 0.0;
    cv::Mat nm, nc;
    kf->predict(m, covariance, nm, nc);
    mean = nm;
    covariance = nc;
}

void SortTrack::multi_predict(std::vector<SortTrack*>& stracks) {
    for (auto* s : stracks) s->predict();
}

void SortTrack::activate(KalmanFilterXYAH* kf_, int frame_id) {
    kf = kf_;
    track_id = next_id();
    cv::Vec4f xyah = convert_coords(_tlwh);
    double z_data[4] = {xyah[0], xyah[1], xyah[2], xyah[3]};
    cv::Mat z = cv::Mat(4, 1, CV_64F, z_data).clone();
    kf->initiate(z, mean, covariance);
    has_kalman = true;
    tracklet_len = 0.f;
    hits = 1;
    state = TrackState::Tracked;
    // Confirmed immediately on the very first frame (official convention),
    // otherwise the track stays tentative until n_init consecutive matches.
    is_activated = (frame_id == 1) || (hits >= n_init);
    this->frame_id = frame_id;
    start_frame = frame_id;
    cv::Vec4f b = xyxy();
    history.push_back(xyxy_center(b));
}

void SortTrack::update(SortTrack* new_track, int frame_id) {
    this->frame_id = frame_id;
    tracklet_len += 1.f;
    hits += 1;
    cv::Vec4f xyah = convert_coords(new_track->tlwh());
    double z_data[4] = {xyah[0], xyah[1], xyah[2], xyah[3]};
    cv::Mat z = cv::Mat(4, 1, CV_64F, z_data).clone();
    cv::Mat nm, nc;
    kf->update(mean, covariance, z, nm, nc);
    mean = nm;
    covariance = nc;
    state = TrackState::Tracked;
    is_activated = is_activated || (hits >= n_init);   // confirm after n_init consecutive hits
    score = new_track->score;
    cls = new_track->cls;
    idx = new_track->idx;
    cv::Vec4f b = xyxy();
    history.push_back(xyxy_center(b));
    if (history.size() > 30) history.erase(history.begin());
}

void SortTrack::re_activate(SortTrack* new_track, int frame_id, bool new_id) {
    cv::Vec4f xyah = convert_coords(new_track->tlwh());
    double z_data[4] = {xyah[0], xyah[1], xyah[2], xyah[3]};
    cv::Mat z = cv::Mat(4, 1, CV_64F, z_data).clone();
    cv::Mat nm, nc;
    kf->update(mean, covariance, z, nm, nc);
    mean = nm;
    covariance = nc;
    tracklet_len = 0.f;
    hits += 1;
    state = TrackState::Tracked;
    is_activated = true;   // a recovered track is output immediately
    this->frame_id = frame_id;
    if (new_id) track_id = next_id();
    score = new_track->score;
    cls = new_track->cls;
    idx = new_track->idx;
    cv::Vec4f b = xyxy();
    history.push_back(xyxy_center(b));
}

// ---------------- OcSortTrack ----------------
static inline cv::Vec4f xyxy2tlwh(const cv::Vec4f& xyxy) {
    return cv::Vec4f(xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]);
}

OcSortTrack::OcSortTrack(const cv::Vec4f& xywh, float score, int cls, int delta_t, int idx)
    : SortTrack(xywh, score, cls, idx), delta_t(delta_t) {}

cv::Point2f OcSortTrack::xyxy_center(const cv::Vec4f& xyxy) {
    return dt::xyxy_center(xyxy);
}

void OcSortTrack::record_observation(const cv::Vec4f& obs, int frame_id) {
    last_observation = obs;
    observations[frame_id] = obs;
    int max_keep = delta_t + 2;
    if (static_cast<int>(observations.size()) > max_keep) {
        for (auto it = observations.begin(); it != observations.end();) {
            if (static_cast<int>(observations.size()) <= max_keep) break;
            it = observations.erase(it);
        }
    }
}

cv::Vec2f OcSortTrack::compute_velocity() {
    if (observations.size() < 2) return cv::Vec2f(0.f, 0.f);
    int cur = observations.rbegin()->first;
    cv::Point2f cur_c = xyxy_center(observations[cur]);
    int prev_frame = -1;
    for (auto it = observations.rbegin(); it != observations.rend(); ++it) {
        if (it->first < cur - delta_t + 1) { prev_frame = it->first; break; }
    }
    cv::Vec4f prev_obs;
    if (prev_frame < 0) {
        int earliest = observations.begin()->first;
        if (earliest == cur) return cv::Vec2f(0.f, 0.f);
        prev_obs = observations[earliest];
    } else {
        prev_obs = observations[prev_frame];
    }
    cv::Point2f prev_c = xyxy_center(prev_obs);
    cv::Vec2f dir(cur_c.x - prev_c.x, cur_c.y - prev_c.y);
    float n = static_cast<float>(std::sqrt(dir.ddot(dir)));
    if (n < 1e-6f) return cv::Vec2f(0.f, 0.f);
    return dir / n;
}

void OcSortTrack::activate(KalmanFilterXYAH* kf_, int frame_id) {
    SortTrack::activate(kf_, frame_id);
    record_observation(xyxy(), frame_id);
    saved_mean = mean.clone();
    saved_cov = covariance.clone();
    velocity = compute_velocity();
}

void OcSortTrack::update(SortTrack* new_track, int frame_id) {
    record_observation(new_track->xyxy(), frame_id);
    SortTrack::update(new_track, frame_id);
    saved_mean = mean.clone();
    saved_cov = covariance.clone();
    velocity = compute_velocity();
}

void OcSortTrack::re_activate(SortTrack* new_track, int frame_id, bool new_id) {
    record_observation(new_track->xyxy(), frame_id);
    SortTrack::re_activate(new_track, frame_id, new_id);
    saved_mean = mean.clone();
    saved_cov = covariance.clone();
    velocity = compute_velocity();
}

void OcSortTrack::apply_oru(const cv::Vec4f& new_obs_xyxy, int current_frame_id) {
    if (saved_mean.empty() || observations.empty()) return;
    int last_frame = observations.rbegin()->first;
    int gap = current_frame_id - last_frame;
    if (gap <= 1) return;

    mean = saved_mean.clone();
    covariance = saved_cov.clone();
    cv::Vec4f last_obs = observations[last_frame];

    for (int t = 1; t < gap; ++t) {
        float alpha = static_cast<float>(t) / static_cast<float>(gap);
        cv::Vec4f virt = (1.0f - alpha) * last_obs + alpha * new_obs_xyxy;
        cv::Vec4f virt_ltwh = xyxy2tlwh(virt);
        cv::Vec4f virt_xyah = tlwh_to_xyah(virt_ltwh);
        double z_data[4] = {virt_xyah[0], virt_xyah[1], virt_xyah[2], virt_xyah[3]};
        cv::Mat z = cv::Mat(4, 1, CV_64F, z_data).clone();
        cv::Mat m, c; kf->predict(mean, covariance, m, c); mean = m; covariance = c;
        cv::Mat m2, c2; kf->update(mean, covariance, z, m2, c2); mean = m2; covariance = c2;
    }
    cv::Mat m, c; kf->predict(mean, covariance, m, c); mean = m; covariance = c;
}

// ---------------- DeepOcSortTrack ----------------
DeepOcSortTrack::DeepOcSortTrack(const cv::Vec4f& xywh, float score, int cls, int delta_t,
                                 const std::vector<float>* feat, float alpha_fixed_emb,
                                 float det_thresh, int idx)
    : OcSortTrack(xywh, score, cls, delta_t, idx),
      alpha_fixed_emb(alpha_fixed_emb), det_thresh(det_thresh) {
    if (feat) update_features(*feat, score);
}

void DeepOcSortTrack::update_features(const std::vector<float>& feat, float score) {
    std::vector<float> f = feat;
    float n = 0.f;
    for (float v : f) n += v * v;
    n = std::sqrt(n);
    if (n < 1e-12f) return;  // zero-norm feature carries no appearance info
    for (float& v : f) v /= n;

    if (!has_feature) {
        curr_feat = f;
        smooth_feat = f;
        has_feature = true;
        return;
    }
    float alpha;
    if (score > det_thresh) {
        float trust = (score - det_thresh) / std::max(1.0f - det_thresh, 1e-9f);
        alpha = alpha_fixed_emb + (1.0f - alpha_fixed_emb) * (1.0f - trust);
    } else {
        alpha = 1.0f;  // low confidence: keep the smoothed feature
    }
    curr_feat = f;
    for (size_t i = 0; i < smooth_feat.size(); ++i)
        smooth_feat[i] = alpha * smooth_feat[i] + (1.0f - alpha) * f[i];
    float sn = 0.f;
    for (float v : smooth_feat) sn += v * v;
    sn = std::sqrt(sn);
    if (sn > 1e-12f) for (float& v : smooth_feat) v /= sn;
}

void DeepOcSortTrack::update(SortTrack* new_track, int frame_id) {
    const std::vector<float>* f = new_track->curr_feature();
    if (f) update_features(*f, new_track->score);
    OcSortTrack::update(new_track, frame_id);
}

void DeepOcSortTrack::re_activate(SortTrack* new_track, int frame_id, bool new_id) {
    const std::vector<float>* f = new_track->curr_feature();
    if (f) update_features(*f, new_track->score);
    OcSortTrack::re_activate(new_track, frame_id, new_id);
}

void DeepOcSortTrack::multi_gmc(std::vector<std::shared_ptr<SortTrack>>& stracks, const cv::Mat& H) {
    std::vector<DeepOcSortTrack*> deep;
    for (auto& s : stracks) {
        SortTrack* d = nullptr;
        s->as_deep(d);
        if (d) deep.push_back(static_cast<DeepOcSortTrack*>(d));
    }
    if (deep.empty()) return;

    int N = static_cast<int>(deep.size());
    cv::Mat means(N, 8, CV_64F);
    for (int i = 0; i < N; ++i) {
        std::memcpy(means.ptr<double>(i), deep[i]->mean.ptr<double>(0), 8 * sizeof(double));
    }

    cv::Mat R = H(cv::Rect(0, 0, 2, 2)).clone();   // 2x2
    cv::Mat t = H(cv::Rect(2, 0, 1, 2)).clone().reshape(1, 2); // 2x1
    cv::Mat R8 = cv::Mat::eye(8, 8, CV_64F);
    R.copyTo(R8(cv::Rect(0, 0, 2, 2)));
    R.copyTo(R8(cv::Rect(4, 4, 2, 2)));

    // means: batched transform is a valid (N x 8) * (8 x 8) product
    cv::Mat means_o = means * R8.t();
    for (int i = 0; i < N; ++i) {
        means_o.at<double>(i, 0) += t.at<double>(0);
        means_o.at<double>(i, 1) += t.at<double>(1);
    }

    for (int i = 0; i < N; ++i) {
        cv::Mat tmp_mean = means_o.row(i).t();
        deep[i]->mean = tmp_mean.clone();                 // 8x1
        // covariance must be transformed per track: cov' = R8 * cov * R8^T.
        // (A vertical stack of N 8x8 blocks is (N*8) x 8 and cannot be
        // left-multiplied by the 8x8 R8 - that was a dimension bug.)
        cv::Mat cov_o = R8 * deep[i]->covariance * R8.t();   // MatExpr -> Mat
        deep[i]->covariance = cov_o.clone();

        if (deep[i]->last_observation[0] >= 0.f) {
            cv::Vec4f obs = deep[i]->last_observation;
            cv::Point2f c = xyxy_center(obs);
            float w = obs[2] - obs[0], h = obs[3] - obs[1];
            double cc_data[2] = {c.x, c.y};
            cv::Mat cc = cv::Mat(2, 1, CV_64F, cc_data).clone();
            cv::Mat nc = R * cc + t;
            deep[i]->last_observation = cv::Vec4f(
                static_cast<float>(nc.at<double>(0)) - w / 2.f,
                static_cast<float>(nc.at<double>(1)) - h / 2.f,
                static_cast<float>(nc.at<double>(0)) + w / 2.f,
                static_cast<float>(nc.at<double>(1)) + h / 2.f);
        }
    }
}

// ---------------- DeepSortTrack (classic DeepSORT) ----------------
namespace {
// L2-normalized copy of a feature; returns false for (near-)zero vectors.
bool normalize_copy(const std::vector<float>& in, std::vector<float>& out) {
    double n2 = 0;
    for (float v : in) n2 += (double)v * v;
    n2 = std::sqrt(n2);
    if (n2 < 1e-12) return false;
    out = in;
    for (float& v : out) v = (float)(v / n2);
    return true;
}
} // namespace

void DeepSortTrack::activate(KalmanFilterXYAH* kf_, int frame_id) {
    OcSortTrack::activate(kf_, frame_id);   // base sets hits=1
    // Paper-faithful DeepSORT has no frame-1 shortcut: a track is tentative
    // until n_init consecutive associations.
    is_activated = (hits >= n_init);
}

void DeepSortTrack::predict() {
    SortTrack::predict();
    time_since_update++;   // age counter for cascade matching
}

void DeepSortTrack::update(SortTrack* new_track, int frame_id) {
    // Gallery append: classic DeepSORT keeps raw normalized features (no EMA);
    // the min-distance over the gallery provides temporal smoothing.
    const std::vector<float>* f = new_track->curr_feature();
    if (f) {
        std::vector<float> g;
        if (normalize_copy(*f, g)) {
            gallery.push_back(g);
            while (gallery.size() > kGalleryCap) gallery.pop_front();
        }
    }
    SortTrack::update(new_track, frame_id);   // KF update + history + hits++ (reused)
    time_since_update = 0;
    is_activated = (hits >= n_init);          // confirm after n_init consecutive hits
}

void DeepSortTrack::re_activate(SortTrack* new_track, int frame_id, bool new_id) {
    const std::vector<float>* f = new_track->curr_feature();
    if (f) {
        std::vector<float> g;
        if (normalize_copy(*f, g)) {
            gallery.push_back(g);
            while (gallery.size() > kGalleryCap) gallery.pop_front();
        }
    }
    SortTrack::re_activate(new_track, frame_id, new_id);   // reused (hits++)
    time_since_update = 0;
    is_activated = (hits >= n_init);   // a previously confirmed track stays confirmed
}

} // namespace dt
