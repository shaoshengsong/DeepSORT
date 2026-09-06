#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <deque>
#include <cmath>
#include <string>
#include "kalman_filter.h"
#include "box_geometry.h"

namespace dt {

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

class KalmanFilterXYAH;  // forward declaration

// ============================================================
// TrackedObjectBase: global unique ID counter and base state
// ============================================================
class TrackedObjectBase {
public:
    static int _count;
    int track_id = 0;
    bool is_activated = false;
    TrackState state = TrackState::New;
    float score = 0.f;
    int start_frame = 0;
    int frame_id = 0;

    static int next_id() { _count++; return _count; }
    static void reset_id() { _count = 0; }
    void mark_lost() { state = TrackState::Lost; }
    void mark_removed() { state = TrackState::Removed; }
};

// ============================================================
// SortTrack: a single target track (XYAH Kalman state)
// ============================================================
class SortTrack : public TrackedObjectBase {
public:
    static KalmanFilterXYAH shared_kalman;  // class-level shared instance (matches reference semantics)

    cv::Vec4f _tlwh;                    // tlwh captured from the first detection
    KalmanFilterXYAH* kf = nullptr;     // points to the shared Kalman instance owned by the tracker
    cv::Mat mean;                      // 8x1 double
    cv::Mat covariance;                 // 8x8 double
    bool has_kalman = false;
    float tracklet_len = 0.f;
    int cls = 0;
    int idx = 0;
    int hits = 0;        // total successful associations (never reset)
    int n_init = 3;      // consecutive matches required before a tentative track is confirmed/output
    float angle = -1.f;                 // -1 means no angle (axis-aligned box)
    std::vector<cv::Point2f> history;  // center history, used to draw the trajectory

    // xywh = (cx, cy, w, h), optional trailing angle; idx is the detection index
    SortTrack(const cv::Vec4f& xywh, float score, int cls, int idx = 0, float angle = -1.f);

    virtual void predict();
    static void multi_predict(std::vector<SortTrack*>& stracks);

    virtual void activate(KalmanFilterXYAH* kf, int frame_id);
    virtual void re_activate(SortTrack* new_track, int frame_id, bool new_id = false);
    virtual void update(SortTrack* new_track, int frame_id);

    cv::Vec4f convert_coords(const cv::Vec4f& tlwh) const;  // tlwh -> xyah
    cv::Vec4f tlwh() const;
    cv::Vec4f xyxy() const;
    cv::Vec4f xywh() const;
    static cv::Vec4f tlwh_to_xyah(const cv::Vec4f& tlwh);
    std::vector<float> result() const;  // [x1,y1,x2,y2,id,score,cls,idx]

    // ---- OC-SORT / Deep OC-SORT virtual hooks (no-op in the base class) ----
    virtual void apply_oru(const cv::Vec4f& new_obs_xyxy, int frame_id) { (void)new_obs_xyxy; (void)frame_id; }
    virtual bool has_velocity() const { return false; }
    virtual cv::Vec2f get_velocity() const { return cv::Vec2f(0.f, 0.f); }
    virtual bool has_observation() const { return false; }
    virtual cv::Vec4f get_last_observation() const { return cv::Vec4f(-1.f, -1.f, -1.f, -1.f); }
    virtual void as_deep(SortTrack*&) { /* base class has no Deep form */ }
    virtual void as_deepsort(SortTrack*&) { /* base class has no classic-DeepSORT form */ }
    virtual const std::vector<float>* smooth_feature() const { return nullptr; }
    virtual const std::vector<float>* curr_feature() const { return nullptr; }
};

// ============================================================
// OcSortTrack: track that records observations (carries the ORU/OCM/OCR triplet)
// ============================================================
class OcSortTrack : public SortTrack {
public:
    cv::Vec4f last_observation = cv::Vec4f(-1.f, -1.f, -1.f, -1.f);
    std::map<int, cv::Vec4f> observations;  // frame_id -> xyxy observation
    cv::Vec2f velocity = cv::Vec2f(0.f, 0.f);
    int delta_t = 3;
    cv::Mat saved_mean, saved_cov;

    OcSortTrack(const cv::Vec4f& xywh, float score, int cls, int delta_t = 3, int idx = 0);

    void activate(KalmanFilterXYAH* kf, int frame_id) override;
    void re_activate(SortTrack* new_track, int frame_id, bool new_id = false) override;
    void update(SortTrack* new_track, int frame_id) override;

    void apply_oru(const cv::Vec4f& new_obs_xyxy, int frame_id) override;
    bool has_velocity() const override { return true; }
    cv::Vec2f get_velocity() const override { return velocity; }
    bool has_observation() const override { return last_observation[0] >= 0.f; }
    cv::Vec4f get_last_observation() const override { return last_observation; }

private:
    void record_observation(const cv::Vec4f& obs, int frame_id);
    cv::Vec2f compute_velocity();
protected:
    static cv::Point2f xyxy_center(const cv::Vec4f& xyxy);
};

// ============================================================
// DeepOcSortTrack: OcSortTrack plus appearance features (ReID + EMA)
// ============================================================
class DeepOcSortTrack : public OcSortTrack {
public:
    std::vector<float> smooth_feat;   // smoothed feature (L2-normalized)
    std::vector<float> curr_feat;     // current-frame feature
    bool has_feature = false;
    float alpha_fixed_emb = 0.95f;
    float det_thresh = 0.25f;

    DeepOcSortTrack(const cv::Vec4f& xywh, float score, int cls, int delta_t = 3,
                    const std::vector<float>* feat = nullptr, float alpha_fixed_emb = 0.95f,
                    float det_thresh = 0.25f, int idx = 0);

    void update(SortTrack* new_track, int frame_id) override;
    void re_activate(SortTrack* new_track, int frame_id, bool new_id = false) override;
    void as_deep(SortTrack*& out) override { out = this; }
    const std::vector<float>* smooth_feature() const override { return has_feature ? &smooth_feat : nullptr; }
    const std::vector<float>* curr_feature() const override { return has_feature ? &curr_feat : nullptr; }

    void update_features(const std::vector<float>& feat, float score);

    // Global motion compensation specialized for XYAH state: rotates only (x,y) and (vx,vy),
    // and syncs last_observation
    static void multi_gmc(std::vector<std::shared_ptr<SortTrack>>& stracks, const cv::Mat& H);

private:
    void record_observation_deep(const cv::Vec4f& obs, int frame_id);
};

// ============================================================
// DeepSortTrack: classic-DeepSORT track. Reuses DeepOcSortTrack's
// feature plumbing (curr_feat / update-re_activate chain) but adds
// the paper's appearance GALLERY (last N features, min cosine
// distance) and the cascade-matching age counter, per Wojke et
// al., ICIP 2017.
// ============================================================
class DeepSortTrack : public DeepOcSortTrack {
public:
    std::deque<std::vector<float>> gallery;  // appearance gallery (latest kGalleryCap)
    int time_since_update = 0;               // frames since last association (cascade level)
    // hits / n_init are inherited from SortTrack (base confirmation machinery)

    using DeepOcSortTrack::DeepOcSortTrack;  // reuse the existing constructor

    void activate(KalmanFilterXYAH* kf, int frame_id) override;             // sets hits=1, confirms iff hits>=n_init
    void predict() override;                                                // + age counter
    void update(SortTrack* new_track, int frame_id) override;               // + gallery append
    void re_activate(SortTrack* new_track, int frame_id, bool new_id = false) override;
    void as_deepsort(SortTrack*& out) override { out = this; }

    static constexpr size_t kGalleryCap = 100;
};

} // namespace dt
