#pragma once
#include <string>

namespace dt {

// Tracker configuration. All per-tracker defaults live in code
// (TrackerConfig::defaults_for). No external YAML/config files are required.
struct TrackerConfig {
    std::string tracker_type = "ocsort";

    // Thresholds
    float track_high_thresh = 0.25f;
    float track_low_thresh = 0.10f;
    float new_track_thresh = 0.25f;
    int   track_buffer = 30;
    float match_thresh = 0.8f;
    float second_match_thresh = 0.5f;  // IoU cost gate for the low-score second association (ByteTrack)
    bool  fuse_score = true;
    int   n_init = 3;   // consecutive matches before a tentative track is confirmed

    // OC-SORT specific
    int   delta_t = 3;
    float inertia = 0.2f;
    bool  use_byte = false;

    // Deep OC-SORT specific
    std::string gmc_method = "none";
    bool  with_reid = false;
    std::string model = "auto";
    float proximity_thresh = 0.5f;   // max IoU DISTANCE (1-IoU) for the appearance gate
    float appearance_thresh = 0.25f; // max cosine DISTANCE for the appearance gate
    float alpha_fixed_emb = 0.95f;

    // Per-tracker defaults. tracker_type selects the preset; recognized keys:
    //   "bytetrack" | "ocsort" | "deepocsort" | "deepsort"
    // Unknown types fall back to the OC-SORT preset.
    static TrackerConfig defaults_for(const std::string& tracker_type);
};

} // namespace dt
