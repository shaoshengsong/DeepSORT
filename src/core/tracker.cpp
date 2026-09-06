#include "tracker.h"
#include "linear_assignment.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <fstream>

namespace dt {

static constexpr float PI = 3.141592653589793f;

// ---------------- Common list helpers ----------------
namespace {
using ST = std::shared_ptr<SortTrack>;

std::vector<ST> joint_stracks(const std::vector<ST>& a, const std::vector<ST>& b) {
    std::vector<ST> out = a;
    std::set<int> ids;
    for (auto& t : a) ids.insert(t->track_id);
    for (auto& t : b) if (ids.find(t->track_id) == ids.end()) out.push_back(t);
    return out;
}
std::vector<ST> sub_stracks(const std::vector<ST>& a, const std::vector<ST>& b) {
    std::set<int> ids;
    for (auto& t : b) ids.insert(t->track_id);
    std::vector<ST> out;
    for (auto& t : a) if (ids.find(t->track_id) == ids.end()) out.push_back(t);
    return out;
}
std::vector<ST> subset(const std::vector<ST>& v, const std::vector<int>& idx) {
    std::vector<ST> out;
    for (int i : idx) out.push_back(v[i]);
    return out;
}
void iou_distance_stracks(const std::vector<ST>& a, const std::vector<ST>& b, std::vector<std::vector<float>>& cost) {
    std::vector<cv::Vec4f> va, vb;
    for (auto& t : a) va.push_back(t->xyxy());
    for (auto& t : b) vb.push_back(t->xyxy());
    iou_distance_matrix(va, vb, cost);
    // Class-aware gating: never associate detections across different classes.
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            if (a[i]->cls != b[j]->cls) cost[i][j] = 1.0f;
}
std::vector<std::vector<float>> fuse_score(const std::vector<std::vector<float>>& cost,
                                           const std::vector<ST>& dets) {
    std::vector<std::vector<float>> out = cost;
    for (size_t i = 0; i < cost.size(); ++i)
        for (size_t j = 0; j < cost[i].size(); ++j)
            out[i][j] = 1.0f - (1.0f - cost[i][j]) * dets[j]->score;
    return out;
}
std::vector<std::vector<float>> embedding_distance(const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    size_t nr = tracks.size(), nc = dets.size();
    std::vector<std::vector<float>> cost(nr, std::vector<float>(nc, 0.f));
    if (nr == 0 || nc == 0) return cost;
    int dim = 0;
    for (auto& t : tracks) { const auto* f = t->smooth_feature(); if (f) { dim = (int)f->size(); break; } }
    for (auto& d : dets) { const auto* f = d->curr_feature(); if (f && (int)f->size() > dim) dim = (int)f->size(); }
    std::vector<std::vector<float>> tf(nr, std::vector<float>(dim, 0.f));
    std::vector<std::vector<float>> df(nc, std::vector<float>(dim, 0.f));
    for (size_t i = 0; i < nr; ++i) { const auto* f = tracks[i]->smooth_feature(); if (f) tf[i] = *f; }
    for (size_t j = 0; j < nc; ++j) { const auto* f = dets[j]->curr_feature(); if (f) df[j] = *f; }
    for (size_t i = 0; i < nr; ++i) {
        for (size_t j = 0; j < nc; ++j) {
            double dot = 0, na = 0, nb = 0;
            for (int k = 0; k < dim; ++k) { dot += tf[i][k] * df[j][k]; na += tf[i][k]*tf[i][k]; nb += df[j][k]*df[j][k]; }
            float c = 1.0f - static_cast<float>(dot / std::max(std::sqrt(na) * std::sqrt(nb), 1e-12));
            cost[i][j] = std::max(0.0f, c);
        }
    }
    // /2 and missing features -> max distance (matches Python semantics)
    std::vector<int> miss_t, miss_d;
    for (size_t i = 0; i < nr; ++i) if (!tracks[i]->smooth_feature()) miss_t.push_back((int)i);
    for (size_t j = 0; j < nc; ++j) if (!dets[j]->curr_feature()) miss_d.push_back((int)j);
    for (auto i : miss_t) for (size_t j = 0; j < nc; ++j) cost[i][j] = 2.0f;
    for (auto j : miss_d) for (size_t i = 0; i < nr; ++i) cost[i][j] = 2.0f;
    for (size_t i = 0; i < nr; ++i) for (size_t j = 0; j < nc; ++j) cost[i][j] /= 2.0f;
    return cost;
}
void remove_duplicate_stracks(std::vector<ST>& at, std::vector<ST>& bt, float dup_thresh = 0.15f) {
    std::vector<std::vector<float>> pd;
    iou_distance_stracks(at, bt, pd);
    std::vector<int> dupa, dupb;
    for (size_t p = 0; p < at.size(); ++p)
        for (size_t q = 0; q < bt.size(); ++q)
            if (pd[p][q] < dup_thresh) {
                int tp = at[p]->frame_id - at[p]->start_frame;
                int tq = bt[q]->frame_id - bt[q]->start_frame;
                if (tp > tq) dupb.push_back((int)q); else dupa.push_back((int)p);
            }
    std::set<int> sa(dupa.begin(), dupa.end()), sb(dupb.begin(), dupb.end());
    std::vector<ST> na, nb;
    for (size_t i = 0; i < at.size(); ++i) if (sa.find((int)i) == sa.end()) na.push_back(at[i]);
    for (size_t i = 0; i < bt.size(); ++i) if (sb.find((int)i) == sb.end()) nb.push_back(bt[i]);
    at.swap(na); bt.swap(nb);
}
} // namespace

// ---------------- BaseTracker: generic MOT pipeline ----------------
BaseTracker::BaseTracker(const TrackerConfig& c)
    : cfg(c), max_frames_lost(c.track_buffer) {
    kalman_filter = KalmanFilterXYAH();
    reset_id();
    reset();
}

// ---------------- ByteTracker: concrete ByteTrack strategy ----------------
ByteTracker::ByteTracker(const TrackerConfig& c) : BaseTracker(c) {
    cfg.use_byte = true;  // ByteTrack hallmark: low-confidence second association
}

void BaseTracker::reset() {
    tracked_stracks.clear();
    lost_stracks.clear();
    removed_stracks.clear();
    removed_stracks_frame.clear();
    frame_id = 0;
    kalman_filter = KalmanFilterXYAH();
    reset_id();
}

void BaseTracker::reset_id() { SortTrack::reset_id(); }

std::pair<std::vector<Detection>, std::vector<Detection>> BaseTracker::split_detections(const FrameDetections& r) const {
    std::vector<Detection> high, low;
    for (size_t i = 0; i < r.size(); ++i) {
        float w = r.xyxy[i][2] - r.xyxy[i][0];
        float h = r.xyxy[i][3] - r.xyxy[i][1];
        if (w <= 0 || h <= 0) continue;
        Detection d;
        d.box = cv::Rect2f(r.xyxy[i][0], r.xyxy[i][1], w, h);
        d.conf = r.conf[i];
        d.cls = r.cls[i];
        d.idx = static_cast<int>(i);
        if (d.conf >= cfg.track_high_thresh) high.push_back(d);
        else if (d.conf > cfg.track_low_thresh && d.conf < cfg.track_high_thresh) low.push_back(d);
    }
    return {high, low};
}

void BaseTracker::split_tracked(std::vector<ST>& unconfirmed, std::vector<ST>& tracked) {
    for (auto& t : tracked_stracks) {
        if (!t->is_activated) unconfirmed.push_back(t);
        else tracked.push_back(t);
    }
}

void BaseTracker::multi_predict(std::vector<ST>& tracks) {
    std::vector<SortTrack*> raw;
    for (auto& t : tracks) raw.push_back(t.get());
    SortTrack::multi_predict(raw);
}

std::vector<ST> BaseTracker::init_track(const std::vector<Detection>& dets, const cv::Mat& aux) {
    (void)aux;
    std::vector<ST> out;
    for (const auto& d : dets) {
        cv::Vec4f xywh(d.box.x + d.box.width / 2.f, d.box.y + d.box.height / 2.f, d.box.width, d.box.height);
        auto t = std::make_shared<SortTrack>(xywh, d.conf, d.cls, d.idx);
        t->n_init = cfg.n_init;   // consecutive matches required before confirmation
        out.push_back(t);
    }
    return out;
}

std::vector<std::vector<float>> BaseTracker::get_dists(const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    std::vector<std::vector<float>> cost;
    iou_distance_stracks(tracks, dets, cost);   // already class-gated
    if (cfg.fuse_score) cost = fuse_score(cost, dets);
    return cost;
}

void BaseTracker::fuse_appearance(std::vector<std::vector<float>>& dists,
                                 const std::vector<ST>& tracks, const std::vector<ST>& dets,
                                 const std::vector<std::vector<float>>* iou_dists) {
    (void)dists; (void)tracks; (void)dets; (void)iou_dists;
}

void BaseTracker::pre_first_associate(std::vector<ST>& pool, std::vector<ST>& unconfirmed,
                                      const cv::Mat& img, const std::vector<Detection>& high) {
    (void)pool; (void)unconfirmed; (void)img; (void)high;
}

std::pair<std::vector<int>, std::vector<int>> BaseTracker::first_association(
    std::vector<ST>& pool, std::vector<ST>& dets, std::vector<ST>& activated, std::vector<ST>& refind) {
    // When the track pool is empty (first frame or right after all tracks are
    // lost), every detection must be treated as unmatched. Otherwise
    // linear_assignment receives a 0-row cost matrix, misreads the column count
    // as 0, and produces an empty u_det.
    if (pool.empty()) {
        std::vector<int> u_det(static_cast<int>(dets.size()));
        for (size_t i = 0; i < dets.size(); ++i) u_det[i] = static_cast<int>(i);
        return {{}, u_det};
    }
    std::vector<std::vector<float>> dists = get_dists(pool, dets);
    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_t, u_d;
    linear_assignment(dists, cfg.match_thresh, matches, u_t, u_d);
    apply_matches(matches, pool, dets, activated, refind);
    return {u_t, u_d};
}

void BaseTracker::apply_match(const ST& track, const ST& det, std::vector<ST>& activated, std::vector<ST>& refind) {
    if (track->state == TrackState::Tracked) {
        track->update(det.get(), frame_id);
        activated.push_back(track);
    } else {
        track->re_activate(det.get(), frame_id, false);
        refind.push_back(track);
    }
}

void BaseTracker::apply_matches(const std::vector<std::pair<int, int>>& matches, std::vector<ST>& pool,
                                std::vector<ST>& dets, std::vector<ST>& activated, std::vector<ST>& refind) {
    for (auto& m : matches) apply_match(pool[m.first], dets[m.second], activated, refind);
}

std::pair<std::vector<int>, std::vector<ST>> BaseTracker::unconfirmed_association(
    std::vector<ST>& unconfirmed, std::vector<int>& u_det, std::vector<ST>& dets,
    std::vector<ST>& activated, std::vector<ST>& removed) {
    std::vector<ST> det_sub;
    for (int i : u_det) det_sub.push_back(dets[i]);
    if (unconfirmed.empty()) {
        std::vector<int> all(det_sub.size());
        for (size_t i = 0; i < det_sub.size(); ++i) all[i] = (int)i;
        return {all, det_sub};
    }
    std::vector<std::vector<float>> dists = get_dists(unconfirmed, det_sub);
    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_unconf, u_d;
    linear_assignment(dists, cfg.match_thresh, matches, u_unconf, u_d);
    for (auto& m : matches) {
        unconfirmed[m.first]->update(det_sub[m.second].get(), frame_id);
        activated.push_back(unconfirmed[m.first]);
    }
    for (int it : u_unconf) {
        unconfirmed[it]->mark_removed();
        removed.push_back(unconfirmed[it]);
    }
    return {u_d, det_sub};
}

void BaseTracker::init_new_tracks(std::vector<int>& u_det, std::vector<ST>& dets,
                                  std::vector<ST>& activated, std::vector<ST>& refind) {
    (void)refind;
    for (int inew : u_det) {
        if (inew < 0 || inew >= (int)dets.size()) continue;
        auto& track = dets[inew];
        if (track->score < cfg.new_track_thresh) continue;
        track->activate(&kalman_filter, frame_id);
        activated.push_back(track);
    }
}

void BaseTracker::remove_stale_lost(std::vector<ST>& removed) {
    for (auto& t : lost_stracks) {
        if (frame_id - t->frame_id > max_frames_lost) {
            t->mark_removed();
            removed.push_back(t);
        }
    }
}

void BaseTracker::merge_track_pools(std::vector<ST>& activated, std::vector<ST>& refind,
                                    std::vector<ST>& lost, std::vector<ST>& removed) {
    // Rebuild: keep only previously Tracked trajectories. Tracks that turned
    // Lost outside second_association are routed into the lost pool here
    // instead of being silently dropped, so they stay re-findable. Pointer
    // identity check avoids double-inserting tracks already collected by
    // second_association.
    {
        std::vector<ST> keep;
        for (auto& t : tracked_stracks) {
            if (t->state == TrackState::Tracked) keep.push_back(t);
            else if (t->state == TrackState::Lost &&
                     std::find(lost.begin(), lost.end(), t) == lost.end())
                lost.push_back(t);
        }
        tracked_stracks = keep;
    }
    tracked_stracks = joint_stracks(tracked_stracks, activated);
    tracked_stracks = joint_stracks(tracked_stracks, refind);
    lost_stracks = sub_stracks(lost_stracks, tracked_stracks);
    for (auto& t : lost) lost_stracks.push_back(t);
    lost_stracks = sub_stracks(lost_stracks, removed_stracks);
    remove_duplicate_stracks(tracked_stracks, lost_stracks);
    removed_stracks_frame = removed;
    for (auto& t : removed) removed_stracks.push_back(t);
    if (removed_stracks.size() > 1000)
        removed_stracks = std::vector<ST>(removed_stracks.end() - 1000, removed_stracks.end());
}

std::vector<TrackOutput> BaseTracker::format_output() {
    std::vector<TrackOutput> out;
    for (auto& t : tracked_stracks) {
        if (!t->is_activated) continue;
        cv::Vec4f b = t->xyxy();
        TrackOutput o;
        o.box = cv::Rect2f(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        o.track_id = t->track_id;
        o.score = t->score;
        o.cls = t->cls;
        o.trajectory = t->history;
        out.push_back(o);
    }
    return out;
}

std::pair<std::vector<int>, std::vector<int>> BaseTracker::post_first_association(
    std::vector<ST>& pool, std::vector<ST>& dets, std::vector<int>& u_track,
    std::vector<int>& u_det, std::vector<ST>& activated, std::vector<ST>& refind) {
    (void)pool; (void)dets; (void)u_track; (void)u_det; (void)activated; (void)refind;
    return {u_track, u_det};
}

void BaseTracker::second_association(std::vector<ST>& pool, std::vector<int>& u_track,
                                     std::vector<ST>& dets_second, std::vector<ST>& activated,
                                     std::vector<ST>& refind, std::vector<ST>& lost) {
    // Low-score second association only runs when use_byte is set; otherwise unmatched tracked targets are marked lost
    if (!cfg.use_byte) {
        for (int i : u_track) {
            auto& t = pool[i];
            if (t->state == TrackState::Tracked) { t->mark_lost(); lost.push_back(t); }
        }
        return;
    }
    std::vector<ST> r_tracked;
    for (int i : u_track) if (pool[i]->state == TrackState::Tracked) r_tracked.push_back(pool[i]);
    if (r_tracked.empty()) return;
    if (!dets_second.empty()) {
        std::vector<std::vector<float>> dists;
        iou_distance_stracks(r_tracked, dets_second, dists);
        std::vector<std::pair<int, int>> matches;
        std::vector<int> u_t, u_d;
        linear_assignment(dists, cfg.second_match_thresh, matches, u_t, u_d);
        apply_matches(matches, r_tracked, dets_second, activated, refind);
        for (int it : u_t) {
            auto& t = r_tracked[it];
            if (t->state != TrackState::Lost) { t->mark_lost(); lost.push_back(t); }
        }
    } else {
        // No low-score detections this frame: every unmatched tracked target
        // must still be marked lost, otherwise it stays Tracked forever
        // (zombie track: never cleaned up, keeps being output and drifting).
        for (auto& t : r_tracked) {
            t->mark_lost();
            lost.push_back(t);
        }
    }
}

std::vector<TrackOutput> BaseTracker::update(const FrameDetections& results, const cv::Mat& img) {
    frame_id++;
    std::vector<ST> activated, refind, lost, removed;

    auto [high, low] = split_detections(results);
    auto detections = init_track(high, img);
    auto detections_second = init_track(low, img);

    std::vector<ST> unconfirmed, tracked;
    split_tracked(unconfirmed, tracked);
    std::vector<ST> strack_pool = joint_stracks(tracked, lost_stracks);
    multi_predict(strack_pool);
    pre_first_associate(strack_pool, unconfirmed, img, high);

    auto [u_track, u_det] = first_association(strack_pool, detections, activated, refind);
    std::tie(u_track, u_det) = post_first_association(strack_pool, detections, u_track, u_det, activated, refind);
    second_association(strack_pool, u_track, detections_second, activated, refind, lost);
    std::tie(u_det, detections) = unconfirmed_association(unconfirmed, u_det, detections, activated, removed);
    init_new_tracks(u_det, detections, activated, refind);
    remove_stale_lost(removed);
    merge_track_pools(activated, refind, lost, removed);
    return format_output();
}

// ---------------- OcSortTracker ----------------
OcSortTracker::OcSortTracker(const TrackerConfig& c) : BaseTracker(c) {
    delta_t = c.delta_t;
    inertia = c.inertia;
    use_byte = c.use_byte;
    cfg.use_byte = use_byte;
}

std::vector<ST> OcSortTracker::init_track(const std::vector<Detection>& dets, const cv::Mat& aux) {
    (void)aux;
    std::vector<ST> out;
    for (const auto& d : dets) {
        cv::Vec4f xywh(d.box.x + d.box.width / 2.f, d.box.y + d.box.height / 2.f, d.box.width, d.box.height);
        auto t = std::make_shared<OcSortTrack>(xywh, d.conf, d.cls, delta_t, d.idx);
        t->n_init = cfg.n_init;   // consecutive matches required before confirmation
        out.push_back(t);
    }
    return out;
}

std::vector<std::vector<float>> OcSortTracker::get_dists(const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    // Official OC-SORT association cost order:
    //   IoU distance  +  inertia * velocity-direction cost   (OCM)
    //   -> optional detection-score fusion
    //   -> optional appearance fusion (Deep OC-SORT)
    // OCM angle cost is arccos(dot)/pi (same scaling as the official
    // implementation; weighted by inertia = 0.2), added on top of IoU.
    std::vector<std::vector<float>> iou;
    iou_distance_stracks(tracks, dets, iou);   // already class-gated
    std::vector<std::vector<float>> dists = iou;
    auto vel = velocity_direction_cost(tracks, dets);
    for (size_t i = 0; i < dists.size(); ++i)
        for (size_t j = 0; j < dists[i].size(); ++j)
            dists[i][j] += inertia * vel[i][j];
    if (cfg.fuse_score) dists = fuse_score(dists, dets);
    fuse_appearance(dists, tracks, dets, &iou);
    // Final class gate: appearance fusion (min) must not revive a cross-class pair.
    for (size_t i = 0; i < tracks.size(); ++i)
        for (size_t j = 0; j < dets.size(); ++j)
            if (tracks[i]->cls != dets[j]->cls) dists[i][j] = 1.0f;
    return dists;
}

std::vector<std::vector<float>> OcSortTracker::velocity_direction_cost(
    const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    size_t nr = tracks.size(), nc = dets.size();
    std::vector<std::vector<float>> cost(nr, std::vector<float>(nc, 0.f));
    if (nr == 0 || nc == 0) return cost;
    std::vector<cv::Point2f> det_centers(nc);
    for (size_t j = 0; j < nc; ++j) det_centers[j] = xyxy_center(dets[j]->xyxy());
    for (size_t i = 0; i < nr; ++i) {
        auto t = std::dynamic_pointer_cast<OcSortTrack>(tracks[i]);
        if (!t || !t->has_velocity()) continue;
        if (t->get_last_observation()[0] < 0) continue;
        cv::Point2f tc = xyxy_center(t->get_last_observation());
        cv::Vec2f v = t->get_velocity();
        for (size_t j = 0; j < nc; ++j) {
            cv::Vec2f dir(det_centers[j].x - tc.x, det_centers[j].y - tc.y);
            float n = static_cast<float>(std::sqrt(dir.ddot(dir)));
            if (n < 1e-6f) continue;
            dir /= n;
            float dot = std::max(-1.f, std::min(1.f, static_cast<float>(dir.ddot(v))));
            cost[i][j] = std::acos(dot) / PI;
        }
    }
    return cost;
}

std::vector<std::vector<float>> OcSortTracker::ocr_distance(const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    std::vector<cv::Vec4f> atl, btl;
    for (auto& t : tracks) {
        auto ot = std::dynamic_pointer_cast<OcSortTrack>(t);
        if (ot && ot->has_observation()) atl.push_back(ot->get_last_observation());
        else atl.push_back(t->xyxy());
    }
    for (auto& d : dets) btl.push_back(d->xyxy());
    std::vector<std::vector<float>> cost;
    iou_distance_matrix(atl, btl, cost);
    // Class-aware gating for observation-centric recovery (OCR).
    for (size_t i = 0; i < tracks.size(); ++i)
        for (size_t j = 0; j < dets.size(); ++j)
            if (tracks[i]->cls != dets[j]->cls) cost[i][j] = 1.0f;
    return cost;
}

std::pair<std::vector<int>, std::vector<int>> OcSortTracker::ocr_associate(
    const std::vector<ST>& tracks, const std::vector<ST>& dets,
    std::vector<ST>& activated, std::vector<ST>& refind) {
    if (tracks.empty() || dets.empty()) {
        std::vector<int> u_t(tracks.size());
        for (size_t i = 0; i < tracks.size(); ++i) u_t[i] = (int)i;
        std::vector<int> u_d(dets.size());
        for (size_t i = 0; i < dets.size(); ++i) u_d[i] = (int)i;
        return {u_t, u_d};
    }
    auto ocr = ocr_distance(tracks, dets);
    auto dists = ocr;
    if (cfg.fuse_score) dists = fuse_score(ocr, dets);
    fuse_appearance(dists, tracks, dets, &ocr);
    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_t, u_d;
    linear_assignment(dists, cfg.match_thresh, matches, u_t, u_d);
    for (auto& m : matches) {
        auto& track = tracks[m.first];
        auto& det = dets[m.second];
        if (track->state == TrackState::Tracked) {
            track->update(det.get(), frame_id);
            activated.push_back(track);
        } else {
            auto ot = std::dynamic_pointer_cast<OcSortTrack>(track);
            if (ot) ot->apply_oru(det->xyxy(), frame_id);
            track->re_activate(det.get(), frame_id, false);
            refind.push_back(track);
        }
    }
    return {u_t, u_d};
}

std::pair<std::vector<int>, std::vector<int>> OcSortTracker::post_first_association(
    std::vector<ST>& pool, std::vector<ST>& dets, std::vector<int>& u_track,
    std::vector<int>& u_det, std::vector<ST>& activated, std::vector<ST>& refind) {
    std::vector<ST> ocr_dets;
    for (int i : u_det) ocr_dets.push_back(dets[i]);
    if (ocr_dets.empty()) return {u_track, u_det};

    std::vector<int> tracked_idx, other_idx;
    for (int k : u_track) {
        if (pool[k]->state == TrackState::Tracked) tracked_idx.push_back(k);
        else other_idx.push_back(k);
    }
    auto [u_t1, u_d1] = ocr_associate(subset(pool, tracked_idx), ocr_dets, activated, refind);
    std::vector<ST> remaining;
    for (int j : u_d1) remaining.push_back(ocr_dets[j]);
    auto [u_t2, u_d2] = ocr_associate(subset(pool, other_idx), remaining, activated, refind);

    std::vector<int> new_u_track, new_u_det;
    for (int t : u_t1) new_u_track.push_back(tracked_idx[t]);
    for (int t : u_t2) new_u_track.push_back(other_idx[t]);
    for (int j : u_d2) new_u_det.push_back(u_det[u_d1[j]]);
    return {new_u_track, new_u_det};
}

void OcSortTracker::second_association(std::vector<ST>& pool, std::vector<int>& u_track,
                                std::vector<ST>& dets_second, std::vector<ST>& activated,
                                std::vector<ST>& refind, std::vector<ST>& lost) {
    if (!use_byte) {
        for (int i : u_track) {
            auto& t = pool[i];
            if (t->state == TrackState::Tracked) { t->mark_lost(); lost.push_back(t); }
        }
        return;
    }
    // use_byte=True: low-score second association (same as base class)
    BaseTracker::second_association(pool, u_track, dets_second, activated, refind, lost);
}

// ---------------- DeepOcSortTracker ----------------
DeepOcSortTracker::DeepOcSortTracker(const TrackerConfig& c, std::shared_ptr<AppearanceEncoder> enc)
    : OcSortTracker(c), gmc(c.gmc_method, 2), encoder(enc) {
    proximity_thresh = c.proximity_thresh;
    appearance_thresh = c.appearance_thresh;
    alpha_fixed_emb = c.alpha_fixed_emb;
}

std::vector<ST> DeepOcSortTracker::init_track(const std::vector<Detection>& dets, const cv::Mat& aux) {
    std::vector<ST> out;
    std::vector<cv::Vec4f> xyxy_vec;
    for (const auto& d : dets) xyxy_vec.push_back(cv::Vec4f(d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height));
    std::vector<std::vector<float>> feats;
    if (encoder && encoder->ok() && !aux.empty()) feats = (*encoder)(aux, xyxy_vec);
    else feats.assign(dets.size(), std::vector<float>());

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        cv::Vec4f xywh(d.box.x + d.box.width / 2.f, d.box.y + d.box.height / 2.f, d.box.width, d.box.height);
        const std::vector<float>* f = feats[i].empty() ? nullptr : &feats[i];
        auto t = std::make_shared<DeepOcSortTrack>(xywh, d.conf, d.cls, delta_t, f,
                                                   alpha_fixed_emb, cfg.new_track_thresh, d.idx);
        t->n_init = cfg.n_init;   // consecutive matches required before confirmation
        out.push_back(t);
    }
    return out;
}

void DeepOcSortTracker::pre_first_associate(std::vector<ST>& pool, std::vector<ST>& unconfirmed,
                                     const cv::Mat& img, const std::vector<Detection>& high) {
    if (img.empty() || cfg.gmc_method == "none" || cfg.gmc_method.empty()) return;
    std::vector<cv::Vec4f> high_xyxy;
    for (const auto& d : high)
        high_xyxy.push_back(cv::Vec4f(d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height));
    cv::Mat warp;
    try { warp = gmc.apply(img, high_xyxy); }
    catch (...) { warp = cv::Mat::eye(2, 3, CV_32F); }
    DeepOcSortTrack::multi_gmc(pool, warp);
    DeepOcSortTrack::multi_gmc(unconfirmed, warp);
}

void DeepOcSortTracker::fuse_appearance(std::vector<std::vector<float>>& dists,
                                 const std::vector<ST>& tracks, const std::vector<ST>& dets,
                                 const std::vector<std::vector<float>>* iou_dists) {
    if (!encoder || !encoder->ok() || tracks.empty() || dets.empty()) return;
    auto emb = embedding_distance(tracks, dets);
    // Both thresholds are DISTANCE thresholds (same semantics as DeepSORT):
    //  - appearance_thresh: max cosine distance (0.25); worse appearance is gated
    //  - proximity_thresh:  max IoU distance (1-IoU = 0.5 <-> IoU 0.5); far boxes are gated
    for (size_t i = 0; i < emb.size(); ++i)
        for (size_t j = 0; j < emb[i].size(); ++j) {
            if (emb[i][j] > appearance_thresh) emb[i][j] = 1.0f;
            if (iou_dists && (*iou_dists)[i][j] > proximity_thresh) emb[i][j] = 1.0f;
            if (tracks[i]->cls != dets[j]->cls) emb[i][j] = 1.0f;
        }
    for (size_t i = 0; i < dists.size(); ++i)
        for (size_t j = 0; j < dists[i].size(); ++j)
            dists[i][j] = std::min(dists[i][j], emb[i][j]);
}

void DeepOcSortTracker::reset() {
    OcSortTracker::reset();
    gmc.reset_params();
}

// ---------------- DeepSortTracker (paper-faithful classic DeepSORT) ----------------
DeepSortTracker::DeepSortTracker(const TrackerConfig& c, std::shared_ptr<AppearanceEncoder> enc)
    : cfg(c), encoder(enc) {
    max_frames_lost = c.track_buffer;
    kalman_filter = KalmanFilterXYAH();
    SortTrack::reset_id();
    reset();
}

void DeepSortTracker::reset() {
    tracked_stracks.clear();
    lost_stracks.clear();
    removed_stracks.clear();
    frame_id = 0;
    kalman_filter = KalmanFilterXYAH();
    SortTrack::reset_id();
}

void DeepSortTracker::multi_predict(std::vector<ST>& tracks) {
    std::vector<SortTrack*> raw;
    for (auto& t : tracks) raw.push_back(t.get());
    SortTrack::multi_predict(raw);
}

std::vector<ST> DeepSortTracker::init_track(const std::vector<Detection>& dets, const cv::Mat& aux) {
    std::vector<ST> out;
    std::vector<cv::Vec4f> xyxy_vec;
    for (const auto& d : dets)
        xyxy_vec.push_back(cv::Vec4f(d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height));
    std::vector<std::vector<float>> feats;
    if (encoder && encoder->ok() && !aux.empty()) feats = (*encoder)(aux, xyxy_vec);
    else feats.assign(dets.size(), std::vector<float>());

    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        cv::Vec4f xywh(d.box.x + d.box.width / 2.f, d.box.y + d.box.height / 2.f, d.box.width, d.box.height);
        const std::vector<float>* f = feats[i].empty() ? nullptr : &feats[i];
        out.push_back(std::make_shared<DeepSortTrack>(xywh, d.conf, d.cls, 1, f,
                                                      0.95f, cfg.new_track_thresh, d.idx));
        // Seed the gallery with the birth feature so the track participates in
        // cascade matching from its second frame on (paper appends on creation).
        {
            SortTrack* s = nullptr;
            out.back()->as_deepsort(s);
            if (auto* ds = static_cast<DeepSortTrack*>(s)) {
                ds->n_init = cfg.n_init;
                if (f) {
                    std::vector<float> g;
                    double n2 = 0;
                    for (float v : *f) n2 += (double)v * v;
                    n2 = std::sqrt(n2);
                    if (n2 > 1e-12) {
                        g = *f;
                        for (float& v : g) v = (float)(v / n2);
                        ds->gallery.push_back(g);
                    }
                }
            }
        }
    }
    return out;
}

std::vector<std::vector<float>> DeepSortTracker::appearance_cost(
    const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    size_t nr = tracks.size(), nc = dets.size();
    std::vector<std::vector<float>> cost(nr, std::vector<float>(nc, 1.0f));
    if (nr == 0 || nc == 0) return cost;
    for (size_t j = 0; j < nc; ++j) {
        const auto* df = dets[j]->curr_feature();
        if (!df || df->empty()) continue;   // detection without features -> stay 1.0
        double dn = 0;
        for (float v : *df) dn += (double)v * v;
        dn = std::sqrt(dn);
        if (dn < 1e-12) continue;
        for (size_t i = 0; i < nr; ++i) {
            // Class-aware gating: never match detections across classes.
            if (tracks[i]->cls != dets[j]->cls) continue;   // cost stays 1.0
            SortTrack* s = nullptr;
            tracks[i]->as_deepsort(s);
            DeepSortTrack* t = static_cast<DeepSortTrack*>(s);
            if (!t || t->gallery.empty()) continue;   // track without features -> 1.0
            // MIN cosine distance over the gallery (paper Eq. appearance term)
            float best = 1.0f;
            for (const auto& g : t->gallery) {
                if (g.size() != df->size()) continue;   // dim mismatch -> skip entry
                double dot = 0, gn = 0;
                for (size_t k = 0; k < g.size(); ++k) { dot += (double)g[k] * (*df)[k]; gn += (double)g[k] * g[k]; }
                gn = std::sqrt(gn);
                if (gn < 1e-12) continue;
                float d_cos = 1.0f - (float)(dot / (gn * dn));
                if (d_cos < best) best = d_cos;
            }
            cost[i][j] = best;
        }
    }
    return cost;
}

std::vector<std::vector<bool>> DeepSortTracker::motion_gate(
    const std::vector<ST>& tracks, const std::vector<ST>& dets) {
    size_t nr = tracks.size(), nc = dets.size();
    std::vector<std::vector<bool>> gate(nr, std::vector<bool>(nc, false));
    for (size_t i = 0; i < nr; ++i) {
        if (!tracks[i]->has_kalman) continue;
        for (size_t j = 0; j < nc; ++j) {
            if (tracks[i]->cls != dets[j]->cls) continue;   // cross-class: gated out
            // detection box -> xyah measurement
            cv::Vec4f xyah = tlwh_to_xyah(dets[j]->tlwh());
            double z_data[4] = {xyah[0], xyah[1], xyah[2], xyah[3]};
            cv::Mat z = cv::Mat(4, 1, CV_64F, z_data).clone();
            gate[i][j] = kalman_filter.mahalanobis2(tracks[i]->mean, tracks[i]->covariance, z) <= kChi2Gate;
        }
    }
    return gate;
}

std::vector<TrackOutput> DeepSortTracker::update(const FrameDetections& results, const cv::Mat& img) {
    frame_id++;
    std::vector<ST> activated, refind, removed;

    // ---- 1. Build detection tracks (with ReID features) ----
    std::vector<Detection> all;
    for (int i = 0; i < results.size(); ++i) {
        float w = results.xyxy[i][2] - results.xyxy[i][0];
        float h = results.xyxy[i][3] - results.xyxy[i][1];
        if (w <= 0 || h <= 0) continue;
        Detection d;
        d.box = cv::Rect2f(results.xyxy[i][0], results.xyxy[i][1], w, h);
        d.conf = results.conf[i];
        d.cls = results.cls[i];
        d.idx = i;
        all.push_back(d);
    }
    auto detections = init_track(all, img);
    const int ndet = (int)detections.size();

    // ---- 2. Predict all active + lost tracks (DeepSortTrack::predict++ ages) ----
    std::vector<ST> pool;
    for (auto& t : tracked_stracks) pool.push_back(t);
    for (auto& t : lost_stracks) pool.push_back(t);
    multi_predict(pool);

    // ---- 3. Drop CONFIRMED tracks that exceeded max age.
    //          Tentative (unconfirmed) tracks are not deleted here — they get
    //          one more chance to match, and are removed in step 7 if they
    //          still miss (paper: tentative tracks die after a single miss). ----
    {
        std::vector<ST> keep;
        for (auto& t : pool) {
            SortTrack* s = nullptr;
            t->as_deepsort(s);
            DeepSortTrack* ds = static_cast<DeepSortTrack*>(s);
            bool confirmed = ds ? ds->is_activated : true;
            int age = ds ? ds->time_since_update : 0;
            if (confirmed && age > max_frames_lost) { t->mark_removed(); removed.push_back(t); }
            else keep.push_back(t);
        }
        pool.swap(keep);
    }

    // ---- 4. Cascade matching: youngest (time_since_update==1,2,...) first.
    //          Per the paper the appearance cascade is restricted to CONFIRMED
    //          tracks; tentative tracks are only recovered by the IoU stage. ----
    std::vector<char> track_matched(pool.size(), 0), det_matched(ndet, 0);
    auto cost = appearance_cost(pool, detections);
    auto gate = motion_gate(pool, detections);
    constexpr float kInfCost = 1e6f;

    for (int age = 1; age <= max_frames_lost; ++age) {
        // tracks at this cascade level: confirmed, not yet matched, exact age
        std::vector<int> level;
        for (int i = 0; i < (int)pool.size(); ++i) {
            if (track_matched[i]) continue;
            SortTrack* s = nullptr;
            pool[i]->as_deepsort(s);
            DeepSortTrack* ds = static_cast<DeepSortTrack*>(s);
            if (ds && ds->is_activated && ds->time_since_update == age) level.push_back(i);
        }
        if (level.empty()) continue;

        // cost matrix for this level (appearance, gated by Mahalanobis)
        std::vector<std::vector<float>> m(level.size(), std::vector<float>(ndet, kInfCost));
        bool any_col = false;
        for (int j = 0; j < ndet; ++j) {
            if (det_matched[j]) continue;
            any_col = true;
            for (size_t li = 0; li < level.size(); ++li) {
                if (gate[level[li]][j])
                    m[li][j] = cost[level[li]][j];   // gated-in: appearance distance
            }
        }
        if (!any_col) continue;

        std::vector<std::pair<int, int>> matches;
        std::vector<int> u_t, u_d;
        linear_assignment(m, cfg.match_thresh, matches, u_t, u_d);
        for (auto& pr : matches) {
            int ti = level[pr.first], dj = pr.second;
            track_matched[ti] = 1;
            det_matched[dj] = 1;
            if (pool[ti]->state == TrackState::Tracked) {
                pool[ti]->update(detections[dj].get(), frame_id);
                activated.push_back(pool[ti]);
            } else {
                pool[ti]->re_activate(detections[dj].get(), frame_id, false);
                refind.push_back(pool[ti]);
            }
        }
    }

    // ---- 5. Second association: IoU for remaining detections.
    //          Per the paper, IoU is only attempted for TENTATIVE tracks and
    //          CONFIRMED tracks lost exactly one frame ago (time_since_update==1).
    //          Tracks lost for longer are handed to the appearance cascade only. ----
    {
        std::vector<int> um_trk;
        for (int i = 0; i < (int)pool.size(); ++i) {
            if (track_matched[i]) continue;
            SortTrack* s = nullptr;
            pool[i]->as_deepsort(s);
            DeepSortTrack* ds = static_cast<DeepSortTrack*>(s);
            bool tentative = ds ? !ds->is_activated : false;
            int age = ds ? ds->time_since_update : 0;
            if (tentative || age == 1) um_trk.push_back(i);
        }
        std::vector<int> um_det;
        for (int j = 0; j < ndet; ++j) if (!det_matched[j]) um_det.push_back(j);
        if (!um_trk.empty() && !um_det.empty()) {
            std::vector<ST> tsub, dsub;
            for (int i : um_trk) tsub.push_back(pool[i]);
            for (int j : um_det) dsub.push_back(detections[j]);
            std::vector<std::vector<float>> iou;
            iou_distance_stracks(tsub, dsub, iou);
            std::vector<std::pair<int, int>> matches;
            std::vector<int> u_t, u_d;
            // paper: max_iou_distance = 0.5  (IoU >= 0.5)
            linear_assignment(iou, 0.5f, matches, u_t, u_d);
            for (auto& pr : matches) {
                int ti = um_trk[pr.first], dj = um_det[pr.second];
                track_matched[ti] = 1;
                det_matched[dj] = 1;
                if (pool[ti]->state == TrackState::Tracked) {
                    pool[ti]->update(detections[dj].get(), frame_id);
                    activated.push_back(pool[ti]);
                } else {
                    pool[ti]->re_activate(detections[dj].get(), frame_id, false);
                    refind.push_back(pool[ti]);
                }
            }
        }
    }

    // ---- 6. Initialize new tracks from remaining high-score detections ----
    for (int j = 0; j < ndet; ++j) {
        if (det_matched[j]) continue;
        if (detections[j]->score < cfg.new_track_thresh) continue;
        detections[j]->activate(&kalman_filter, frame_id);
        activated.push_back(detections[j]);
    }

    // ---- 7. Rebuild pools ----
    // The pool ALREADY contains both the previous tracked and lost tracks (they
    // were merged in step 2), so the lost list must be rebuilt from the pool
    // alone. Walking the old lost_stracks again would duplicate tracks, which
    // made time_since_update grow exponentially and delete tracks early.
    tracked_stracks = activated;   // this frame's matched tracks
    for (auto& t : refind) tracked_stracks.push_back(t);
    std::set<int> matched_ids;
    for (auto& t : tracked_stracks) matched_ids.insert(t->track_id);
    std::set<int> removed_ids;
    for (auto& t : removed) removed_ids.insert(t->track_id);

    std::vector<ST> new_lost;
    for (int i = 0; i < (int)pool.size(); ++i) {
        if (track_matched[i]) continue;                       // went into tracked
        auto& t = pool[i];
        if (removed_ids.count(t->track_id)) continue;         // expired this frame
        // Tentative (unconfirmed) tracks that missed this frame are deleted
        // immediately (paper: a tentative track dies after a single miss).
        SortTrack* s = nullptr;
        t->as_deepsort(s);
        DeepSortTrack* ds = static_cast<DeepSortTrack*>(s);
        if (ds && !ds->is_activated) {
            t->mark_removed();
            removed.push_back(t);
            continue;
        }
        if (t->state == TrackState::Tracked) t->mark_lost();
        if (t->state == TrackState::Lost) new_lost.push_back(t);
    }
    // dedupe by identity (shared_ptr) in case a track appears twice in the pool
    {
        std::set<ST> seen;
        std::vector<ST> dd;
        for (auto& t : new_lost)
            if (seen.insert(t).second) dd.push_back(t);
        new_lost.swap(dd);
    }
    lost_stracks.swap(new_lost);
    remove_duplicate_stracks(tracked_stracks, lost_stracks);
    for (auto& t : removed) removed_stracks.push_back(t);
    if (removed_stracks.size() > 1000)
        removed_stracks = std::vector<ST>(removed_stracks.end() - 1000, removed_stracks.end());

    // ---- 8. Output confirmed (activated) tracks only ----
    std::vector<TrackOutput> out;
    for (auto& t : tracked_stracks) {
        if (!t->is_activated) continue;
        cv::Vec4f b = t->xyxy();
        TrackOutput o;
        o.box = cv::Rect2f(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        o.track_id = t->track_id;
        o.score = t->score;
        o.cls = t->cls;
        o.trajectory = t->history;
        out.push_back(o);
    }
    return out;
}

// ---------------- Factory ----------------
namespace {
// Build the ReID encoder when the preset requires it. Returns nullptr when
// the model is unavailable/fails to load, so appearance trackers gracefully
// degrade to motion-only (or the caller falls back to a motion tracker).
std::shared_ptr<AppearanceEncoder> make_encoder(const TrackerConfig& cfg, const std::string& model_dir) {
    if (!cfg.with_reid || cfg.model.empty() || cfg.model == "auto") return nullptr;
    std::string path = cfg.model;
    if (!std::ifstream(path)) {
        std::string alt = model_dir + "/" + cfg.model;
        if (std::ifstream(alt)) path = alt;
        else return nullptr;   // model file missing -> motion-only
    }
    auto enc = std::make_shared<AppearanceEncoder>(path);
    if (!enc->ok()) return nullptr;
    return enc;
}
} // namespace

std::shared_ptr<ITracker> create_tracker(const TrackerConfig& cfg, const std::string& model_dir) {
    if (cfg.tracker_type == "deepocsort") {
        // ReID on by default; a missing model silently degrades to OC-SORT motion.
        auto enc = make_encoder(cfg, model_dir);
        return std::make_shared<DeepOcSortTracker>(cfg, enc);
    }
    if (cfg.tracker_type == "deepsort") {
        auto enc = make_encoder(cfg, model_dir);
        if (!enc) {
            // DeepSORT is meaningless without appearance; fall back to ByteTrack.
            TrackerConfig fb = TrackerConfig::defaults_for("bytetrack");
            return std::make_shared<ByteTracker>(fb);
        }
        return std::make_shared<DeepSortTracker>(cfg, enc);
    }
    if (cfg.tracker_type == "bytetrack")
        return std::make_shared<ByteTracker>(cfg);  // use_byte enabled by the strategy itself
    return std::make_shared<OcSortTracker>(cfg);
}

} // namespace dt
