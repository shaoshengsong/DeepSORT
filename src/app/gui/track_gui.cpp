// DeepTracker GUI front-end (Win32 dialog)
// ------------------------------------------------------------
// Core tracking algorithm by: shaoshengsong
// Video detection + (Deep) OC-SORT tracking, with video selection,
// optional tracking, live preview, and double-click-to-run (models and
// configs sit next to the exe or fall back to known paths).
// ------------------------------------------------------------
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <fstream>

#include <opencv2/opencv.hpp>

#include "track_gui.h"
#include "detector/yolo_detector.h"
#include "core/tracker.h"
#include "core/tracker_config.h"

using namespace dt;

// ---------------- Global state ----------------
static HWND g_hDlg = nullptr;
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_cancel{false};

struct RunParams {
    std::string video;
    std::string out;
    bool        tracking = true;
    float       conf = 0.25f;
    int         cls = 0;
    bool        preview = true;
    std::string det_model;
    std::string reid_model;
};
static RunParams g_params;

// ---------------- Path auto-resolution ----------------
static bool file_exists(const std::string& p) {
    std::ifstream f(p);
    return (bool)f;
}

static std::string first_existing(const std::vector<std::string>& cands) {
    for (auto& c : cands) if (file_exists(c)) return c;
    return "";
}

static std::string module_dir() {
    char buf[MAX_PATH] = {0};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string s(buf);
    size_t pos = s.find_last_of("\\/");
    return (pos == std::string::npos) ? "." : s.substr(0, pos);
}

static void resolve_paths(RunParams& p) {
    std::string d = module_dir();
    std::vector<std::string> det_cands = {
        d + "/models/yolo26s.onnx",
        d + "/yolo26s.onnx",
        d + "/../../models/yolo26s.onnx",
        d + "/../../../models/yolo26s.onnx",
    };
    std::vector<std::string> reid_cands = {
        d + "/yolo26s-reid.onnx",
        d + "/models/yolo26s-reid.onnx",
        d + "/../yolo26s-reid.onnx",
        d + "/../../models/yolo26s-reid.onnx",
        d + "/../../../models/yolo26s-reid.onnx",
    };
    p.det_model  = first_existing(det_cands);
    p.reid_model = first_existing(reid_cands);
}

// ---------------- Log/progress helpers ----------------
static void gui_log(const char* msg) {
    char* p = _strdup(msg);
    PostMessage(g_hDlg, WM_GUI_LOG, 0, (LPARAM)p);
}
static void gui_prog(const char* msg) {
    char* p = _strdup(msg);
    PostMessage(g_hDlg, WM_GUI_PROG, 0, (LPARAM)p);
}
static void gui_done(int code) {
    PostMessage(g_hDlg, WM_GUI_DONE, (WPARAM)code, 0);
}

static std::string default_out_path(const std::string& video) {
    size_t dot = video.find_last_of('.');
    if (dot == std::string::npos) return video + "_tracked.mp4";
    return video.substr(0, dot) + "_tracked.mp4";
}

// ---------------- Worker thread ----------------
static DWORD WINAPI WorkerProc(LPVOID) {
    const RunParams& p = g_params;

    YoloDetector detector;
    detector.opts.imgsz = 640;
    detector.opts.conf_thresh = p.conf;
    detector.opts.iou_thresh = 0.45f;
    detector.opts.target_class = p.cls;
    detector.opts.filter_class = (p.cls >= 0);

    if (!detector.load(p.det_model)) {
        gui_log(("[ERROR] Failed to load detection model: " + p.det_model + "\n").c_str());
        gui_done(1);
        return 0;
    }
    gui_log(("[OK] Detector ready: " + p.det_model + "\n").c_str());

    std::shared_ptr<ITracker> tracker;
    if (p.tracking) {
        // All tracker parameters are compiled in (TrackerConfig::defaults_for).
        TrackerConfig cfg = TrackerConfig::defaults_for("deepocsort");
        gui_log("[OK] Tracker config: compiled-in defaults (deepocsort)\n");
        // ReID is enabled for Deep OC-SORT and DeepSORT when the model is available.
        bool reid_ok = (cfg.tracker_type == "deepocsort" || cfg.tracker_type == "deepsort")
                       && !p.reid_model.empty() && file_exists(p.reid_model);
        if (reid_ok) {
            auto enc = std::make_shared<AppearanceEncoder>(p.reid_model);
            if (enc->ok()) {
                tracker = std::make_shared<DeepOcSortTracker>(cfg, enc);
                gui_log("[OK] Tracker: Deep OC-SORT (with ReID)\n");
            } else {
                tracker = std::make_shared<OcSortTracker>(cfg);
                gui_log("[OK] Tracker: OC-SORT (ReID load failed)\n");
            }
        } else if (cfg.tracker_type == "ocsort" || cfg.tracker_type == "deepocsort") {
            tracker = std::make_shared<OcSortTracker>(cfg);
            gui_log("[OK] Tracker: OC-SORT\n");
        } else {
            tracker = std::make_shared<ByteTracker>(cfg);
            gui_log("[OK] Tracker: ByteTrack\n");
        }
    } else {
        gui_log("[INFO] Tracking disabled, drawing detections only\n");
    }

    cv::VideoCapture cap(p.video);
    if (!cap.isOpened()) {
        gui_log(("[ERROR] Cannot open input: " + p.video + "\n").c_str());
        gui_done(1);
        return 0;
    }
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    gui_log(("[INFO] Input " + std::to_string(W) + "x" + std::to_string(H) +
             " @ " + std::to_string((int)fps) + "fps\n").c_str());

    std::string out = p.out.empty() ? default_out_path(p.video) : p.out;
    cv::VideoWriter writer;
    writer.open(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H), true);
    if (!writer.isOpened()) gui_log("[WARN] Failed to open output video, preview only\n");

    bool preview_ok = p.preview;
    if (preview_ok) {
        try { cv::namedWindow("DeepTracker Preview", cv::WINDOW_AUTOSIZE); }
        catch (...) { preview_ok = false; }
    }

    cv::Mat frame;
    int frame_no = 0, max_id = 0;
    while (cap.read(frame)) {
        if (frame.empty() || g_cancel.load()) break;
        ++frame_no;

        auto dets = detector.detect(frame);

        cv::Mat vis = frame.clone();
        if (p.tracking && tracker) {
            FrameDetections fd;
            fd.xyxy.reserve(dets.size());
            fd.conf.reserve(dets.size());
            fd.cls.reserve(dets.size());
            for (auto& d : dets) {
                fd.xyxy.push_back(cv::Vec4f(d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height));
                fd.conf.push_back(d.conf);
                fd.cls.push_back(d.cls);
            }
            auto tracks = tracker->update(fd, frame);
            for (auto& t : tracks) {
                int x = (int)t.box.x, y = (int)t.box.y;
                int w = (int)t.box.width, h = (int)t.box.height;
                cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 200, 255), 2);
                if (t.trajectory.size() >= 2) {
                    for (size_t k = 1; k < t.trajectory.size(); ++k)
                        cv::line(vis, t.trajectory[k - 1], t.trajectory[k], cv::Scalar(0, 255, 0), 1);
                }
                cv::putText(vis, "ID:" + std::to_string(t.track_id), cv::Point(x, y - 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                if (t.track_id > max_id) max_id = t.track_id;
            }
            if (frame_no % 10 == 0)
                gui_prog(("Processing frame " + std::to_string(frame_no) + " | active tracks " +
                          std::to_string((int)tracks.size()) + "\n").c_str());
        } else {
            for (auto& d : dets) {
                int x = (int)d.box.x, y = (int)d.box.y;
                int w = (int)d.box.width, h = (int)d.box.height;
                cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 200, 255), 2);
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", d.conf);
                cv::putText(vis, buf, cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 255), 2);
            }
            if (frame_no % 10 == 0)
                gui_prog(("Detecting frame " + std::to_string(frame_no) + " | objects " +
                          std::to_string((int)dets.size()) + "\n").c_str());
        }

        cv::putText(vis, "frame " + std::to_string(frame_no), cv::Point(10, 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        if (writer.isOpened()) writer.write(vis);
        if (preview_ok) {
            try {
                cv::imshow("DeepTracker Preview", vis);
                int k = cv::waitKey(1);
                if (k == 27) { g_cancel.store(true); break; }
            } catch (...) { preview_ok = false; }
        }
    }

    cap.release();
    if (writer.isOpened()) writer.release();
    if (preview_ok) cv::destroyAllWindows();

    if (g_cancel.load())
        gui_log(("[STOP] Stopped, processed " + std::to_string(frame_no) + " frames\n").c_str());
    else
        gui_log(("[DONE] Processed " + std::to_string(frame_no) + " frames, max track_id=" +
                 std::to_string(max_id) + ", output: " + out + "\n").c_str());

    gui_done(0);
    return 0;
}

// ---------------- Dialog procedure ----------------
static void AppendLog(HWND h, const char* msg) {
    HWND log = GetDlgItem(h, IDC_LOG);
    int len = GetWindowTextLength(log);
    SendMessage(log, EM_SETSEL, (WPARAM)len, (LPARAM)len);
    SendMessageA(log, EM_REPLACESEL, 0, (LPARAM)msg);
}

static INT_PTR CALLBACK DlgProc(HWND h, UINT msg, WPARAM w, LPARAM l) {
    switch (msg) {
        case WM_INITDIALOG: {
            g_hDlg = h;
            SendDlgItemMessage(h, IDC_TRACK_CHK, BM_SETCHECK, BST_CHECKED, 0);
            SendDlgItemMessage(h, IDC_PREVIEW_CHK, BM_SETCHECK, BST_CHECKED, 0);
            SetDlgItemTextA(h, IDC_CONF_EDIT, "0.25");
            SetDlgItemTextA(h, IDC_CLASS_EDIT, "0");
            SetDlgItemTextA(h, IDC_PROGRESS, "Ready");
            EnableWindow(GetDlgItem(h, IDC_STOP_BTN), FALSE);

            resolve_paths(g_params);
            std::string info = "[INFO] Detection model: " + (g_params.det_model.empty() ? std::string("(not found)")
                                                                                     : g_params.det_model);
            gui_log(info.c_str());
            g_params.tracking = true;
            g_params.preview = true;
            g_params.conf = 0.25f;
            g_params.cls = 0;
            return TRUE;
        }
        case WM_COMMAND: {
            int id = LOWORD(w);
            if (id == IDC_VIDEO_BTN) {
                char buf[MAX_PATH] = {0};
                OPENFILENAMEA ofn = {0};
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = h;
                ofn.lpstrFilter = "Video files\0*.mp4;*.avi;*.mkv;*.mov;*.flv;*.wmv\0All files\0*.*\0";
                ofn.lpstrFile = buf;
                ofn.nMaxFile = MAX_PATH;
                ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
                if (GetOpenFileNameA(&ofn)) {
                    SetDlgItemTextA(h, IDC_VIDEO_EDIT, buf);
                    // Auto-fill output path (only if the user has not edited it yet)
                    char outbuf[MAX_PATH];
                    GetDlgItemTextA(h, IDC_OUT_EDIT, outbuf, MAX_PATH);
                    if (outbuf[0] == '\0') SetDlgItemTextA(h, IDC_OUT_EDIT, default_out_path(buf).c_str());
                }
            } else if (id == IDC_OUT_BTN) {
                char buf[MAX_PATH] = {0};
                OPENFILENAMEA ofn = {0};
                ofn.lStructSize = sizeof(ofn);
                ofn.hwndOwner = h;
                ofn.lpstrFilter = "MP4 video\0*.mp4\0AVI video\0*.avi\0All files\0*.*\0";
                ofn.lpstrFile = buf;
                ofn.nMaxFile = MAX_PATH;
                ofn.Flags = OFN_OVERWRITEPROMPT;
                if (GetSaveFileNameA(&ofn)) SetDlgItemTextA(h, IDC_OUT_EDIT, buf);
            } else if (id == IDC_RUN_BTN) {
                if (g_running.load()) break;
                char vbuf[MAX_PATH], obuf[MAX_PATH], cbuf[32], clbuf[32];
                GetDlgItemTextA(h, IDC_VIDEO_EDIT, vbuf, MAX_PATH);
                GetDlgItemTextA(h, IDC_OUT_EDIT, obuf, MAX_PATH);
                GetDlgItemTextA(h, IDC_CONF_EDIT, cbuf, 32);
                GetDlgItemTextA(h, IDC_CLASS_EDIT, clbuf, 32);
                if (vbuf[0] == '\0') { MessageBoxA(h, "Please select a video file first", "Hint", MB_OK); break; }

                g_params.video = vbuf;
                g_params.out = obuf;
                g_params.tracking = (SendDlgItemMessage(h, IDC_TRACK_CHK, BM_GETCHECK, 0, 0) == BST_CHECKED);
                g_params.preview = (SendDlgItemMessage(h, IDC_PREVIEW_CHK, BM_GETCHECK, 0, 0) == BST_CHECKED);
                g_params.conf = (float)atof(cbuf);
                g_params.cls = atoi(clbuf);
                resolve_paths(g_params);

                g_running.store(true);
                g_cancel.store(false);
                SetDlgItemTextA(h, IDC_PROGRESS, "Running...");
                EnableWindow(GetDlgItem(h, IDC_RUN_BTN), FALSE);
                EnableWindow(GetDlgItem(h, IDC_STOP_BTN), TRUE);

                // Parameters are already filled into the global g_params above; start the thread
                CreateThread(nullptr, 0, WorkerProc, nullptr, 0, nullptr);
            } else if (id == IDC_STOP_BTN) {
                if (g_running.load()) {
                    g_cancel.store(true);
                    SetDlgItemTextA(h, IDC_PROGRESS, "Stopping...");
                }
            }
            return TRUE;
        }
        case WM_GUI_LOG: {
            if (l) { AppendLog(h, (char*)l); free((void*)l); }
            return TRUE;
        }
        case WM_GUI_PROG: {
            if (l) { SetDlgItemTextA(h, IDC_PROGRESS, (char*)l); free((void*)l); }
            return TRUE;
        }
        case WM_GUI_DONE: {
            g_running.store(false);
            EnableWindow(GetDlgItem(h, IDC_RUN_BTN), TRUE);
            EnableWindow(GetDlgItem(h, IDC_STOP_BTN), FALSE);
            SetDlgItemTextA(h, IDC_PROGRESS, (w == 0) ? "Done" : "Error");
            return TRUE;
        }
        case WM_CLOSE:
            if (g_running.load()) {
                g_cancel.store(true);
                return TRUE; // Ignore close while running; wait for the thread to finish
            }
            EndDialog(h, 0);
            return TRUE;
    }
    return FALSE;
}

int WINAPI WinMain(HINSTANCE h, HINSTANCE, LPSTR, int) {
    return (int)DialogBoxParamA(h, MAKEINTRESOURCEA(IDD_MAIN), nullptr, DlgProc, 0);
}
