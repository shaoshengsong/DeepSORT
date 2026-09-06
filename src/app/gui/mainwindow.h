#pragma once
#include <QMainWindow>
#include <QThread>
#include <QString>
#include <QImage>
#include <atomic>
#include <string>
#include <memory>
#include <vector>
#include <QPushButton>
#include <QLabel>
#include <QList>
#include <QAction>

#include <opencv2/opencv.hpp>

#include "core/tracker.h"
#include "core/tracker_config.h"
#include "detector/yolo_detector.h"

// Parameters for a single run
struct GuiRunParams {
    QString video;
    QString out;
    QString trackerType;   // key: "ocsort"/"deepocsort"/"bytetrack"/"deepsort"/"detect"
    float   conf   = 0.25f;
    int     cls    = 0;
    // ReID is auto-selected: enabled iff a valid reidModel path is set
    // (auto-found on disk or picked by the user); leave empty to disable.
    QString gmcMethod = "none";
    QString detModel, reidModel;
};

// Worker thread: runs detection/tracking in the background and sends logs,
// progress and preview frames back to the main thread via signals.
class TrackerWorker : public QThread {
    Q_OBJECT
public:
    explicit TrackerWorker(QObject* parent = nullptr) : QThread(parent) {}
    void setParams(const GuiRunParams& p) { params_ = p; }
    void requestStop() { stopRequested_.store(true); }

signals:
    void logMsg(const QString&);
    void frameReady(const QImage&);
    // Real inference progress: current frame, total frames (0 = unknown),
    // and a short status note ("轨迹 5" / "目标 12")
    void progress(int frame, int total, const QString& note);
    void finished(int);

protected:
    void run() override;

private:
    GuiRunParams params_;
    std::atomic<bool> stopRequested_{false};
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    bool nativeEvent(const QByteArray& eventType, void* message, qintptr* result) override;
    void closeEvent(QCloseEvent* e) override;

private slots:
    void onBrowseVideo();
    void onBrowseOut();
    void onBrowseReid();
    void onTrackerTypeChanged(int);
    void onRun();
    void onStop();
    void appendLog(const QString&);
    void showFrame(const QImage&);
    void onProgress(int frame, int total, const QString& note);
    void onFinished(int code);
    void onThemeSelected(QAction* action);

private:
    QWidget* buildUi();
    void resolvePaths();
    void log(const QString&);

    // ---------- 无边框窗口：拖拽 / 缩放 ----------
    static constexpr int RESIZE_BORDER = 6;   // 边缘热区宽度（逻辑像素）
    static constexpr int TITLE_H       = 34;  // 顶部可拖拽标题条高度（逻辑像素）

    // ---------- 换肤系统 ----------
    struct Theme {
        QString name;          // 主题名称
        QString primaryColor;  // 主色调（按钮、进度条、强调）
        QString lightColor;    // 浅色变体（窗口背景）
        QString darkColor;     // 深色变体（标题条背景）
        QString accentColor;   // 强调色（hover 等）
    };
    void initThemes();
    void applyTheme(int index);
    void buildThemeMenu();

    QList<Theme> m_themes;
    int m_currentTheme = 5;   // 默认=深邃青

    TrackerWorker* worker_ = nullptr;

    // 无边框标题条
    QWidget*     m_central   = nullptr;
    QWidget*     m_title     = nullptr;
    QLabel*      m_titleText = nullptr;
    QPushButton* m_themeBtn  = nullptr;
    QPushButton* m_minBtn    = nullptr;
    QPushButton* m_closeBtn  = nullptr;

    // 功能控件
    class QLineEdit*     edVideo_    = nullptr;
    class QLineEdit*     edOut_      = nullptr;
    class QLineEdit*     edReid_     = nullptr;
    class QLineEdit*     edConf_     = nullptr;
    class QLineEdit*     edClass_    = nullptr;
    class QPushButton*   btnVideo_   = nullptr;
    class QPushButton*   btnOut_     = nullptr;
    class QPushButton*   btnReid_    = nullptr;
    class QPushButton*   btnRun_     = nullptr;
    class QPushButton*   btnStop_    = nullptr;
    class QComboBox*     cmbTracker_ = nullptr;
    class QComboBox*     cmbGmc_     = nullptr;
    class QLabel*        lblReidHint_ = nullptr;
    class QLabel*        lblPreview_ = nullptr;
    class QProgressBar*  prog_       = nullptr;
    class QPlainTextEdit* logEdit_   = nullptr;
    class QLabel*        lblAbout_   = nullptr;   // 底部 GitHub 链接

    GuiRunParams params_;
};
