#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include "tracking_types.h"
#include "kalman_filter.h"
#include "tracked_object.h"
#include "tracker_config.h"
#include "global_motion_compensation.h"
#include "reid/appearance_encoder.h"
#include "box_geometry.h"

namespace dt {

// ============================================================
// ITracker: minimal public interface shared by every tracking
// algorithm. It is intentionally tiny (update / reset / frame_id)
// — it is NOT the pipeline. The pipeline lives in BaseTracker,
// while DeepSortTracker implements its own loop and only agrees
// on this thin contract so callers can treat all algorithms
// uniformly.
// ============================================================
class ITracker {
public:
    int frame_id = 0;
    virtual ~ITracker() = default;
    virtual std::vector<TrackOutput> update(const FrameDetections& results, const cv::Mat& img = cv::Mat()) = 0;
    virtual void reset() = 0;
};

// ============================================================
// BaseTracker: generic multi-object tracking pipeline (the shared
// update() skeleton). Concrete algorithms — ByteTrack, OC-SORT,
// Deep OC-SORT — are different policy combinations on this
// pipeline, injected through virtual hooks.
// ============================================================
class BaseTracker : public ITracker {
public:
    TrackerConfig cfg;
    int max_frames_lost = 30;
    KalmanFilterXYAH kalman_filter;
    std::vector<std::shared_ptr<SortTrack>> tracked_stracks, lost_stracks, removed_stracks, removed_stracks_frame;

    explicit BaseTracker(const TrackerConfig& c);
    virtual ~BaseTracker() = default;

    // Called every frame: takes this frame's detections (xyxy) and an optional
    // current-frame image (used for GlobalMotionCompensation / ReID).
    // Virtual: classic DeepSORT replaces the whole loop with its own.
    virtual std::vector<TrackOutput> update(const FrameDetections& results, const cv::Mat& img = cv::Mat());

    virtual void reset();
    static void reset_id();

protected:
    // ---- Hooks that subclasses may override ----
    virtual std::vector<std::shared_ptr<SortTrack>> init_track(const std::vector<Detection>& dets, const cv::Mat& aux);
    virtual std::vector<std::vector<float>> get_dists(const std::vector<std::shared_ptr<SortTrack>>& tracks,
                                                      const std::vector<std::shared_ptr<SortTrack>>& dets);
    virtual void pre_first_associate(std::vector<std::shared_ptr<SortTrack>>& pool,
                                      std::vector<std::shared_ptr<SortTrack>>& unconfirmed,
                                      const cv::Mat& img,
                                      const std::vector<Detection>& high);
    virtual std::pair<std::vector<int>, std::vector<int>> post_first_association(
        std::vector<std::shared_ptr<SortTrack>>& pool,
        std::vector<std::shared_ptr<SortTrack>>& dets,
        std::vector<int>& u_track,
        std::vector<int>& u_det,
        std::vector<std::shared_ptr<SortTrack>>& activated,
        std::vector<std::shared_ptr<SortTrack>>& refind);
    virtual void second_association(std::vector<std::shared_ptr<SortTrack>>& pool,
                                    std::vector<int>& u_track,
                                    std::vector<std::shared_ptr<SortTrack>>& dets_second,
                                    std::vector<std::shared_ptr<SortTrack>>& activated,
                                    std::vector<std::shared_ptr<SortTrack>>& refind,
                                    std::vector<std::shared_ptr<SortTrack>>& lost);
    virtual void fuse_appearance(std::vector<std::vector<float>>& dists,
                                 const std::vector<std::shared_ptr<SortTrack>>& tracks,
                                 const std::vector<std::shared_ptr<SortTrack>>& dets,
                                 const std::vector<std::vector<float>>* iou_dists);

    // ---- Common helpers ----
    std::pair<std::vector<Detection>, std::vector<Detection>> split_detections(const FrameDetections& r) const;
    void split_tracked(std::vector<std::shared_ptr<SortTrack>>& unconfirmed,
                       std::vector<std::shared_ptr<SortTrack>>& tracked);
    std::pair<std::vector<int>, std::vector<int>> first_association(
        std::vector<std::shared_ptr<SortTrack>>& pool,
        std::vector<std::shared_ptr<SortTrack>>& dets,
        std::vector<std::shared_ptr<SortTrack>>& activated,
        std::vector<std::shared_ptr<SortTrack>>& refind);
    void apply_match(const std::shared_ptr<SortTrack>& track, const std::shared_ptr<SortTrack>& det,
                     std::vector<std::shared_ptr<SortTrack>>& activated,
                     std::vector<std::shared_ptr<SortTrack>>& refind);
    void apply_matches(const std::vector<std::pair<int, int>>& matches,
                       std::vector<std::shared_ptr<SortTrack>>& pool,
                       std::vector<std::shared_ptr<SortTrack>>& dets,
                       std::vector<std::shared_ptr<SortTrack>>& activated,
                       std::vector<std::shared_ptr<SortTrack>>& refind);
    std::pair<std::vector<int>, std::vector<std::shared_ptr<SortTrack>>> unconfirmed_association(
        std::vector<std::shared_ptr<SortTrack>>& unconfirmed,
        std::vector<int>& u_det,
        std::vector<std::shared_ptr<SortTrack>>& dets,
        std::vector<std::shared_ptr<SortTrack>>& activated,
        std::vector<std::shared_ptr<SortTrack>>& removed);
    virtual void init_new_tracks(std::vector<int>& u_det,
                         std::vector<std::shared_ptr<SortTrack>>& dets,
                         std::vector<std::shared_ptr<SortTrack>>& activated,
                         std::vector<std::shared_ptr<SortTrack>>& refind);
    void remove_stale_lost(std::vector<std::shared_ptr<SortTrack>>& removed);
    std::vector<TrackOutput> format_output();
    void merge_track_pools(std::vector<std::shared_ptr<SortTrack>>& activated,
                           std::vector<std::shared_ptr<SortTrack>>& refind,
                           std::vector<std::shared_ptr<SortTrack>>& lost,
                           std::vector<std::shared_ptr<SortTrack>>& removed);
    virtual void multi_predict(std::vector<std::shared_ptr<SortTrack>>& tracks);
};

// ============================================================
// ByteTracker: the concrete ByteTrack strategy (Zhang et al.,
// ECCV 2022) on the BaseTracker pipeline = high-score first
// association + low-confidence second association (BYTE).
// Motion only; the low-score pass is always enabled.
// ============================================================
class ByteTracker : public BaseTracker {
public:
    explicit ByteTracker(const TrackerConfig& c);
};

// ============================================================
// OcSortTracker: implements the ORU / OCM / OCR triplet on top of BaseTracker
// ============================================================
class OcSortTracker : public BaseTracker {
public:
    int delta_t = 3;
    float inertia = 0.2f;
    bool use_byte = false;

    explicit OcSortTracker(const TrackerConfig& c);

protected:
    std::vector<std::shared_ptr<SortTrack>> init_track(const std::vector<Detection>& dets, const cv::Mat& aux) override;
    std::vector<std::vector<float>> get_dists(const std::vector<std::shared_ptr<SortTrack>>& tracks,
                                              const std::vector<std::shared_ptr<SortTrack>>& dets) override;
    std::pair<std::vector<int>, std::vector<int>> post_first_association(
        std::vector<std::shared_ptr<SortTrack>>& pool,
        std::vector<std::shared_ptr<SortTrack>>& dets,
        std::vector<int>& u_track,
        std::vector<int>& u_det,
        std::vector<std::shared_ptr<SortTrack>>& activated,
        std::vector<std::shared_ptr<SortTrack>>& refind) override;
    void second_association(std::vector<std::shared_ptr<SortTrack>>& pool,
                            std::vector<int>& u_track,
                            std::vector<std::shared_ptr<SortTrack>>& dets_second,
                            std::vector<std::shared_ptr<SortTrack>>& activated,
                            std::vector<std::shared_ptr<SortTrack>>& refind,
                            std::vector<std::shared_ptr<SortTrack>>& lost) override;

    // OC-SORT specific helpers
    std::vector<std::vector<float>> velocity_direction_cost(
        const std::vector<std::shared_ptr<SortTrack>>& tracks,
        const std::vector<std::shared_ptr<SortTrack>>& dets);
    std::vector<std::vector<float>> ocr_distance(
        const std::vector<std::shared_ptr<SortTrack>>& tracks,
        const std::vector<std::shared_ptr<SortTrack>>& dets);
    std::pair<std::vector<int>, std::vector<int>> ocr_associate(
        const std::vector<std::shared_ptr<SortTrack>>& tracks,
        const std::vector<std::shared_ptr<SortTrack>>& dets,
        std::vector<std::shared_ptr<SortTrack>>& activated,
        std::vector<std::shared_ptr<SortTrack>>& refind);
};

// ============================================================
// DeepOcSortTracker: OC-SORT + appearance features (ReID) + GlobalMotionCompensation
//                     + adaptive fusion
// ============================================================
class DeepOcSortTracker : public OcSortTracker {
public:
    GlobalMotionCompensation gmc;
    float proximity_thresh = 0.5f;    // max IoU DISTANCE (1-IoU) for the appearance gate
    float appearance_thresh = 0.25f;  // max cosine DISTANCE for the appearance gate
    float alpha_fixed_emb = 0.95f;
    std::shared_ptr<AppearanceEncoder> encoder;

    DeepOcSortTracker(const TrackerConfig& c, std::shared_ptr<AppearanceEncoder> enc);

protected:
    std::vector<std::shared_ptr<SortTrack>> init_track(const std::vector<Detection>& dets, const cv::Mat& aux) override;
    void pre_first_associate(std::vector<std::shared_ptr<SortTrack>>& pool,
                             std::vector<std::shared_ptr<SortTrack>>& unconfirmed,
                             const cv::Mat& img,
                             const std::vector<Detection>& high) override;
    void fuse_appearance(std::vector<std::vector<float>>& dists,
                         const std::vector<std::shared_ptr<SortTrack>>& tracks,
                         const std::vector<std::shared_ptr<SortTrack>>& dets,
                         const std::vector<std::vector<float>>* iou_dists) override;
    void reset() override;
};

// ============================================================
// DeepSortTracker: paper-faithful classic DeepSORT
// (Wojke et al., ICIP 2017). It does NOT use the BaseTracker
// pipeline — it owns its own main loop:
//   - cascade matching by time_since_update (most recently
//     associated tracks match first),
//   - appearance cost = MIN cosine distance over the track's
//     feature gallery (last 100),
//   - squared-Mahalanobis motion gate (chi2, 4 dof, 0.95 -> 9.4877),
//   - second-stage IoU association for remaining detections,
//   - confirmed-track output only.
// It only shares the thin ITracker contract with the pipeline.
// Requires a ReID encoder.
// ============================================================
class DeepSortTracker : public ITracker {
public:
    TrackerConfig cfg;
    int max_frames_lost = 30;
    KalmanFilterXYAH kalman_filter;
    std::vector<std::shared_ptr<SortTrack>> tracked_stracks, lost_stracks, removed_stracks;
    std::shared_ptr<AppearanceEncoder> encoder;
    // Chi-square (0.95, 4 dof) gating threshold for squared Mahalanobis distance.
    static constexpr double kChi2Gate = 9.4877;

    DeepSortTracker(const TrackerConfig& c, std::shared_ptr<AppearanceEncoder> enc);

    // Paper-faithful main loop.
    std::vector<TrackOutput> update(const FrameDetections& results, const cv::Mat& img = cv::Mat()) override;
    void reset() override;

private:
    std::vector<std::shared_ptr<SortTrack>> init_track(const std::vector<Detection>& dets, const cv::Mat& aux);
    void multi_predict(std::vector<std::shared_ptr<SortTrack>>& tracks);

    // Appearance cost matrix: min cosine distance over each track's gallery.
    // Missing features (either side) yield the max cost 1.0.
    std::vector<std::vector<float>> appearance_cost(
        const std::vector<std::shared_ptr<SortTrack>>& tracks,
        const std::vector<std::shared_ptr<SortTrack>>& dets);
    // Squared-Mahalanobis gating mask: true where the detection is inside
    // the track's predicted observation ellipsoid.
    std::vector<std::vector<bool>> motion_gate(
        const std::vector<std::shared_ptr<SortTrack>>& tracks,
        const std::vector<std::shared_ptr<SortTrack>>& dets);
};

// Factory: create a tracker from a config; model_dir resolves relative ReID model paths.
std::shared_ptr<ITracker> create_tracker(const TrackerConfig& cfg, const std::string& model_dir = ".");

} // namespace dt
