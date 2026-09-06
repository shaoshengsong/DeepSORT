#include "tracker_config.h"

namespace dt {

TrackerConfig TrackerConfig::defaults_for(const std::string& tracker_type) {
    TrackerConfig c;
    c.tracker_type = tracker_type;

    if (tracker_type == "bytetrack") {
        // ByteTrack: two-stage association, motion only
        c.track_high_thresh = 0.25f;
        c.track_low_thresh  = 0.10f;
        c.new_track_thresh  = 0.25f;
        c.track_buffer      = 30;
        c.match_thresh      = 0.8f;   // high-score + unconfirmed association gate
        c.second_match_thresh = 0.5f; // low-score second association gate (official default)
        c.fuse_score        = true;
        c.n_init            = 3;
        c.use_byte          = true;   // ByteTrack enables the low-score second pass
    }
    else if (tracker_type == "deepocsort") {
        // Deep OC-SORT: OC-SORT + ReID appearance fusion + GMC.
        // ReID is on by default (the encoder is auto-created when the model
        // is present; trackers gracefully degrade to motion-only if not);
        // thresholds are DISTANCE thresholds, matching the DeepSORT scale.
        c.track_high_thresh = 0.3f;
        c.track_low_thresh  = 0.1f;
        c.new_track_thresh  = 0.3f;
        c.track_buffer      = 30;
        c.match_thresh      = 0.8f;
        c.second_match_thresh = 0.5f;
        c.fuse_score        = true;
        c.n_init            = 3;
        c.delta_t           = 3;
        c.inertia           = 0.2f;
        c.use_byte          = false;
        c.gmc_method        = "none";   // sparseOptFlow|orb|sift|ecc|none; enable for moving-camera scenes
        c.with_reid         = true;
        c.model             = "yolo26s-reid.onnx";
        c.proximity_thresh  = 0.5f;     // max IoU distance (1-IoU): box overlap gate for appearance
        c.appearance_thresh = 0.25f;    // max cosine distance: appearance gate (same scale as DeepSORT)
        c.alpha_fixed_emb   = 0.95f;
    }
    else if (tracker_type == "deepsort") {
        // Paper-faithful DeepSORT (Wojke et al., ICIP 2017); requires ReID.
        // Cascaded appearance matching (min cosine over a 100-frame gallery)
        // gated by a squared-Mahalanobis motion gate (chi2, 4 dof = 9.4877),
        // followed by an IoU second association for tentative + just-lost tracks.
        c.track_high_thresh = 0.3f;
        c.track_low_thresh  = 0.1f;
        c.new_track_thresh  = 0.3f;
        c.track_buffer      = 30;      // max age for CONFIRMED tracks
        c.match_thresh      = 0.2f;    // max cosine distance in cascade matching (paper: 0.2)
        c.fuse_score        = false;   // DeepSORT uses pure appearance distance, no score fusion
        c.n_init            = 3;       // consecutive matches to confirm a tentative track
        c.with_reid         = true;
        c.model             = "yolo26s-reid.onnx";
    }
    // else: OC-SORT preset = the struct's built-in defaults
    return c;
}

} // namespace dt
