# ============================================================
# DeepTracker Qt GUI  (Qt 6.9.3 MSVC2022 64bit)
# Open this .pro in Qt Creator to build (select the
# "Desktop Qt 6.9.3 MSVC2022 64bit" kit).
# Core tracking algorithm by: shaoshengsong
# ------------------------------------------------------------
# This .pro compiles the core + detector + reid sources directly
# and does NOT depend on any CMake build artifacts. All paths
# are relative to $$PWD, so the whole folder is portable: copy
# it anywhere and it still builds. The executable lands in
# $$ROOT/build, which already contains onnxruntime.dll /
# opencv_world500.dll / ffmpeg dll / models / configs, so the
# built exe runs by double-click.
# ============================================================

QT       += core gui widgets
CONFIG   += c++17
CONFIG   -= app_bundle

# Qt6/msvc required flags (strict MSVC conformance; avoids
# binding rvalues to non-const references, etc.)
QMAKE_CXXFLAGS += /permissive- /Zc:__cplusplus

# ---- Root dir: this .pro lives in src/app/gui, so three
#     levels up reaches the project root. ----
ROOT = $$PWD/../../..
SRC  = $$ROOT/src
EXT  = $$ROOT/external

TARGET    = track_gui_qt
# Executable output dir = project root / build, sharing the
# runtime dll / models / configs with the CMake build.
DESTDIR   = $$ROOT/build

# ---- Dependency paths (third-party libs live under external/) ----
OPENCV_DIR = $$EXT/opencv/build
ORT_DIR    = $$EXT/onnxruntime

INCLUDEPATH += \
    $$OPENCV_DIR/include \
    $$ORT_DIR/include \
    $$SRC \
    $$SRC/core \
    $$SRC/detector \
    $$SRC/reid \
    $$PWD

LIBS += -L$$OPENCV_DIR/x64/vc16/lib -lopencv_world500
LIBS += -L$$ORT_DIR/lib -lonnxruntime
# nativeEvent frameless window: GetWindowRect / GetDpiForWindow live in user32
win32:LIBS += -luser32

# ---- Sources: compile core + detector + reid directly so the
#     .pro is self-contained. ----
SOURCES += \
    $$SRC/core/tracker_config.cpp \
    $$SRC/core/global_motion_compensation.cpp \
    $$SRC/core/kalman_filter.cpp \
    $$SRC/core/lap_solver.cpp \
    $$SRC/core/linear_assignment.cpp \
    $$SRC/reid/appearance_encoder.cpp \
    $$SRC/core/tracked_object.cpp \
    $$SRC/core/tracker.cpp \
    $$SRC/detector/yolo_detector.cpp \
    $$PWD/mainwindow.cpp \
    $$PWD/main.cpp

HEADERS += $$PWD/mainwindow.h

# UI icons (embedded via Qt resource system)
RESOURCES += $$PWD/icons.qrc

# ---- After build, deploy Qt runtimes (Qt6Core/Gui/Widgets +
#     platforms plugin) so the exe runs standalone. DESTDIR
#     already holds onnxruntime.dll / opencv_world500.dll /
#     ffmpeg dll / models / configs. If windeployqt is not on
#     PATH, fall back with || echo so the build still succeeds. ----
win32 {
    QMAKE_POST_LINK += $$shell_path($$[QT_INSTALL_BINS]/windeployqt) $$shell_path($$DESTDIR/$$TARGET.exe) || echo "windeployqt skipped (copy Qt dlls manually)"
}
