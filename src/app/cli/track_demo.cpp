// DeepTracker demo: YOLO detection + (Deep) OC-SORT / DeepSORT tracking.
// See PrintUsage() for usage. By default it reads people_walking.mp4 from the
// working directory, runs detection with yolo26s.onnx and, for the appearance
// trackers (Deep OC-SORT / DeepSORT), uses yolo26s-reid.onnx, then writes an
// annotated video.
//
// Build via the CMakeLists.txt in this directory; at runtime place
// yolo26s.onnx / yolo26s-reid.onnx on an accessible path (--det / --reid).
// All tracker parameters are compiled in (TrackerConfig::defaults_for).

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include "detector/yolo_detector.h"
#include "core/tracker.h"
#include "core/tracker_config.h"

using namespace dt;

static void PrintUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --det     <path>  Detection ONNX model (default: models/yolo26s.onnx)\n"
        << "  --reid    <path>  ReID ONNX model (default: yolo26s-reid.onnx; falls back to motion-only if absent)\n"
        << "  --tracker <name>  Tracker preset: bytetrack|ocsort|deepocsort|deepsort (default: deepocsort)\n"
        << "  --source  <path>  Input video / image sequence (default: people_walking.mp4)\n"
        << "  --out     <path>  Output video (default: track_output.mp4)\n"
        << "  --imgsz   <int>   Inference size (default 640)\n"
        << "  --conf    <float> Detection confidence threshold (default 0.25)\n"
        << "  --class   <int>   Track class id; -1 disables filtering (default 0=person)\n"
        << "  --show             Show a live preview window\n";
}

static bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return (bool)f;
}

static std::shared_ptr<ITracker> make_tracker(const TrackerConfig& cfg,
                                                 const std::string& reid_path) {
    // ReID is enabled for Deep OC-SORT and DeepSORT when the ReID model file is present.
    bool needs_reid = (cfg.tracker_type == "deepocsort" || cfg.tracker_type == "deepsort");
    std::string rp = (cfg.model != "auto") ? cfg.model : reid_path;
    bool reid_available = needs_reid && file_exists(rp);
    std::shared_ptr<AppearanceEncoder> enc;
    if (reid_available) {
        enc = std::make_shared<AppearanceEncoder>(rp);
        if (!enc->ok()) {
            std::cerr << "[WARN] Failed to load ReID model, falling back to motion-only: " << rp << "\n";
            enc = nullptr;
            reid_available = false;
        }
    }
    if (cfg.tracker_type == "deepsort") {
        if (reid_available) return std::make_shared<DeepSortTracker>(cfg, enc);
        std::cerr << "[WARN] DeepSORT requires a ReID model; falling back to ByteTrack\n";
        return std::make_shared<ByteTracker>(cfg);
    }
    if (reid_available) return std::make_shared<DeepOcSortTracker>(cfg, enc);
    if (cfg.tracker_type == "ocsort" || cfg.tracker_type == "deepocsort")
        return std::make_shared<OcSortTracker>(cfg);
    return std::make_shared<ByteTracker>(cfg);
}

int main(int argc, char** argv) {
    std::string det_path = "models/yolo26s.onnx";
    std::string reid_path = "yolo26s-reid.onnx";
    std::string tracker_kind = "deepocsort";
    std::string source = "people_walking.mp4";
    std::string out_path = "track_output.mp4";
    int imgsz = 640;
    float conf = 0.25f;
    int cls = 0;
    bool show = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing argument: " << name << "\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--det") det_path = need("--det");
        else if (a == "--reid") reid_path = need("--reid");
        else if (a == "--tracker") tracker_kind = need("--tracker");
        else if (a == "--source") source = need("--source");
        else if (a == "--out") out_path = need("--out");
        else if (a == "--imgsz") imgsz = std::stoi(need("--imgsz"));
        else if (a == "--conf") conf = std::stof(need("--conf"));
        else if (a == "--class") cls = std::stoi(need("--class"));
        else if (a == "--show") show = true;
        else if (a == "-h" || a == "--help") { PrintUsage(argv[0]); return 0; }
        else { std::cerr << "Unknown argument: " << a << "\n"; PrintUsage(argv[0]); return 2; }
    }

    // ---- Detector ----
    YoloDetector detector;
    detector.opts.imgsz = imgsz;
    detector.opts.conf_thresh = conf;
    detector.opts.iou_thresh = 0.45f;
    detector.opts.target_class = cls;
    detector.opts.filter_class = (cls >= 0);
    if (!detector.load(det_path)) {
        std::cerr << "[ERROR] Failed to load detection model: " << det_path << "\n";
        return 1;
    }
    std::cout << "[OK] Detector ready: " << det_path << "\n";

    // ---- Tracker configuration ----
    // All per-tracker parameters are compiled in (TrackerConfig::defaults_for).
    TrackerConfig cfg = TrackerConfig::defaults_for(tracker_kind);
    std::cout << "[OK] Tracker config: type=" << cfg.tracker_type
              << " gmc=" << cfg.gmc_method << " with_reid=" << cfg.with_reid << "\n";
    auto tracker = make_tracker(cfg, reid_path);
    std::cout << "[OK] Tracker ready (tracker_id=" << tracker->frame_id << ")\n";

    // ---- Input ----
    cv::VideoCapture cap(source);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open input: " << source << "\n";
        return 1;
    }
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    std::cout << "[INFO] Input resolution " << W << "x" << H << " @ " << fps << "fps\n";

    cv::VideoWriter writer;
    if (!out_path.empty()) {
        writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H), true);
        if (!writer.isOpened()) {
            std::cerr << "[WARN] Failed to open output video, preview/print only\n";
        }
    }

    cv::Mat frame;
    int frame_no = 0;
    int max_id = 0;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        ++frame_no;

        // Detection
        auto dets = detector.detect(frame);

        if (frame_no <= 3) {
            for (size_t i = 0; i < dets.size() && i < 4; ++i)
                std::cout << "   d" << i << " conf=" << dets[i].conf << " cls=" << dets[i].cls
                          << " box=(" << dets[i].box.x << "," << dets[i].box.y << ","
                          << dets[i].box.width << "," << dets[i].box.height << ")\n";
        }

        // Convert to tracker input
        FrameDetections fd;
        fd.xyxy.reserve(dets.size());
        fd.conf.reserve(dets.size());
        fd.cls.reserve(dets.size());
        for (auto& d : dets) {
            fd.xyxy.push_back(cv::Vec4f(d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height));
            fd.conf.push_back(d.conf);
            fd.cls.push_back(d.cls);
        }

        // Tracking
        auto tracks = tracker->update(fd, frame);

        // Rendering
        cv::Mat vis = frame.clone();
        for (auto& t : tracks) {
            int x = (int)t.box.x, y = (int)t.box.y;
            int w = (int)t.box.width, h = (int)t.box.height;
            cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 200, 255), 2);
            // Trajectory
            if (t.trajectory.size() >= 2) {
                for (size_t k = 1; k < t.trajectory.size(); ++k) {
                    cv::line(vis, t.trajectory[k - 1], t.trajectory[k], cv::Scalar(0, 255, 0), 1);
                }
            }
            std::string label = "ID:" + std::to_string(t.track_id);
            cv::putText(vis, label, cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2);
            if (t.track_id > max_id) max_id = t.track_id;
        }
        cv::putText(vis, "frame " + std::to_string(frame_no) + " tracks " + std::to_string((int)tracks.size()),
                    cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        if (writer.isOpened()) writer.write(vis);
        if (show) {
            cv::imshow("DeepTracker", vis);
            if (cv::waitKey(1) == 27) break;  // ESC to quit
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    if (show) cv::destroyAllWindows();
    std::cout << "[DONE] Processed " << frame_no << " frames, max track_id=" << max_id
              << ", output: " << (out_path.empty() ? "(none)" : out_path) << "\n";
    return 0;
}
