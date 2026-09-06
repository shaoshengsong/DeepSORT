// DeepTracker Qt GUI front-end (frameless + theme + real progress bar)
// Core tracking algorithm by: shaoshengsong
#include "mainwindow.h"

#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QProgressBar>
#include <QPlainTextEdit>
#include <QFileDialog>
#include <QFile>
#include <QCloseEvent>
#include <QScrollBar>
#include <QSettings>
#include <QMenu>
#include <QFileInfo>
#include <QElapsedTimer>
#include <QPainter>
#include <QPen>
#include <QPainterPath>

#ifdef Q_OS_WIN
#  include <windows.h>
#  include <windowsx.h>   // GET_X_LPARAM / GET_Y_LPARAM
#endif

using namespace dt;

// cv::Mat (BGR) -> QImage (RGB, deep copy, thread-safe)
static QImage matToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();
    cv::Mat rgb;
    if (mat.channels() == 3)      cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else if (mat.channels() == 1) cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else                          rgb = mat.clone();
    QImage img(rgb.data, rgb.cols, rgb.rows, (int)rgb.step[0], QImage::Format_RGB888);
    return img.copy();
}

// Resolve a resource relative to the executable (portable: the whole folder can be
// copied anywhere and still be found). Searches dir / dir/.. / dir/../.. for
// <sub>/<name>, returns the absolute path, or empty if not found.
static QString findResource(const QString& dir, const QString& sub, const QString& name) {
    QStringList bases = { dir, dir + "/..", dir + "/../.." };
    for (const QString& b : bases) {
        QString p = b + "/" + sub + "/" + name;
        if (QFile::exists(p)) return QFileInfo(p).absoluteFilePath();
    }
    return QString();
}

// ---- Title-bar glyphs drawn in code: crisp white strokes on transparency,
//      guaranteed visible on any dark theme (better than downloaded icon
//      bitmaps that carry their own filled background square). ----
enum class TitleGlyph { Minimize, Close };

static QIcon titleBarIcon(TitleGlyph kind) {
    const int S = 32;   // 2x for HiDPI sharpness
    QPixmap pm(S, S);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing, true);
    QPen pen(QColor(255, 255, 255), 2.4);
    pen.setCapStyle(Qt::RoundCap);
    p.setPen(pen);
    if (kind == TitleGlyph::Minimize) {
        p.drawLine(QPointF(S * 0.22, S * 0.62), QPointF(S * 0.78, S * 0.62));
    } else {
        const qreal m = S * 0.26;
        p.drawLine(QPointF(m, m), QPointF(S - m, S - m));
        p.drawLine(QPointF(S - m, m), QPointF(m, S - m));
    }
    p.end();
    return QIcon(pm);
}

// ---------------- Worker thread: detection + tracking ----------------
void TrackerWorker::run() {
    stopRequested_.store(false);
    const GuiRunParams& p = params_;
    QElapsedTimer timer;
    timer.start();

    YoloDetector detector;
    detector.opts.imgsz         = 640;
    detector.opts.conf_thresh   = p.conf;
    detector.opts.iou_thresh    = 0.45f;
    detector.opts.target_class  = p.cls;
    detector.opts.filter_class  = (p.cls >= 0);

    if (!detector.load(p.detModel.toStdString())) {
        emit logMsg(tr("[ERROR] Failed to load detection model: %1").arg(p.detModel));
        emit finished(1);
        return;
    }
    emit logMsg(tr("[OK] Detection model ready: %1").arg(p.detModel));

    std::shared_ptr<ITracker> tracker;
    bool tracking = (p.trackerType != "detect");
    if (tracking) {
        // All tracker parameters are compiled in (TrackerConfig::defaults_for).
        TrackerConfig cfg = TrackerConfig::defaults_for(p.trackerType.toStdString());

        // Build the ReID encoder once (auto: enabled iff a valid model is set;
        // used by Deep OC-SORT / DeepSORT, silently ignored by the others)
        auto makeEncoder = [&]() -> std::shared_ptr<AppearanceEncoder> {
            if (!p.reidModel.isEmpty() && QFile::exists(p.reidModel)) {
                auto enc = std::make_shared<AppearanceEncoder>(p.reidModel.toStdString());
                if (enc->ok()) return enc;
                emit logMsg(tr("[WARN] Failed to load ReID model: %1").arg(p.reidModel));
            }
            return nullptr;
        };

        if (p.trackerType == "deepocsort") {
            auto enc = makeEncoder();
            if (enc) {
                cfg.gmc_method = p.gmcMethod.toStdString();
                tracker = std::make_shared<DeepOcSortTracker>(cfg, enc);
                emit logMsg(tr("[OK] Tracker: Deep OC-SORT (ReID, GMC=%1)").arg(p.gmcMethod));
            } else {
                tracker = std::make_shared<OcSortTracker>(cfg);
                emit logMsg(tr("[WARN] ReID unavailable, falling back to OC-SORT"));
            }
        }
        else if (p.trackerType == "deepsort") {
            auto enc = makeEncoder();
            if (enc) {
                tracker = std::make_shared<DeepSortTracker>(cfg, enc);
                emit logMsg(tr("[OK] Tracker: DeepSORT (cascade matching + feature gallery + Mahalanobis gating)"));
            } else {
                tracker = std::make_shared<ByteTracker>(cfg);
                emit logMsg(tr("[WARN] DeepSORT requires ReID; no valid model found, falling back to ByteTrack"));
            }
        }
        else if (p.trackerType == "bytetrack") {
            tracker = std::make_shared<ByteTracker>(cfg);
            emit logMsg(tr("[OK] Tracker: ByteTrack"));
        }
        else {  // ocsort
            tracker = std::make_shared<OcSortTracker>(cfg);
            emit logMsg(tr("[OK] Tracker: OC-SORT"));
        }
    } else {
        emit logMsg(tr("[INFO] Detection-only mode (tracking disabled)"));
    }

    cv::VideoCapture cap(p.video.toStdString());
    if (!cap.isOpened()) {
        emit logMsg(tr("[ERROR] Cannot open video: %1").arg(p.video));
        emit finished(1);
        return;
    }
    int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    // Total frame count drives the real progress bar (0 = unknown, busy mode)
    int totalFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (totalFrames <= 0) totalFrames = 0;
    emit logMsg(tr("[INFO] Input %1x%2 @ %3 fps, %4 frames")
                .arg(W).arg(H).arg((int)fps).arg(totalFrames > 0 ? QString::number(totalFrames) : tr("unknown")));
    emit progress(0, totalFrames, QString());

    std::string out = p.out.toStdString();
    cv::VideoWriter writer;
    if (!out.empty()) {
        writer.open(out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(W, H), true);
        if (!writer.isOpened()) emit logMsg(tr("[WARN] Failed to create output video (preview still available)"));
    }

    cv::Mat frame;
    int frame_no = 0, max_id = 0;
    while (cap.read(frame)) {
        if (frame.empty() || stopRequested_.load()) break;
        ++frame_no;

        auto dets = detector.detect(frame);
        cv::Mat vis = frame.clone();
        QString note;

        if (tracking && tracker) {
            FrameDetections fd;
            fd.xyxy.reserve(dets.size());
            fd.conf.reserve(dets.size());
            fd.cls.reserve(dets.size());
            for (auto& d : dets) {
                fd.xyxy.push_back(cv::Vec4f(d.box.x, d.box.y,
                                            d.box.x + d.box.width, d.box.y + d.box.height));
                fd.conf.push_back(d.conf);
                fd.cls.push_back(d.cls);
            }
            auto tracks = tracker->update(fd, frame);
            for (auto& t : tracks) {
                int x = (int)t.box.x, y = (int)t.box.y;
                int w = (int)t.box.width, h = (int)t.box.height;
                cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 200, 255), 2);
                if (t.trajectory.size() >= 2)
                    for (size_t k = 1; k < t.trajectory.size(); ++k)
                        cv::line(vis, t.trajectory[k - 1], t.trajectory[k], cv::Scalar(0, 255, 0), 1);
                cv::putText(vis, "ID:" + std::to_string(t.track_id),
                            cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 255), 2);
                if (t.track_id > max_id) max_id = t.track_id;
            }
            note = tr("Tracks %1").arg((int)tracks.size());
        } else {
            for (auto& d : dets) {
                int x = (int)d.box.x, y = (int)d.box.y;
                int w = (int)d.box.width, h = (int)d.box.height;
                cv::rectangle(vis, cv::Rect(x, y, w, h), cv::Scalar(0, 200, 255), 2);
                cv::putText(vis, QString::number(d.conf, 'f', 2).toStdString(),
                            cv::Point(x, y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 255), 2);
            }
            note = tr("Objects %1").arg((int)dets.size());
        }

        cv::putText(vis, ("frame " + std::to_string(frame_no)).c_str(),
                    cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);

        if (writer.isOpened()) writer.write(vis);

        // Real progress: current frame / total frames, every frame
        emit progress(frame_no, totalFrames, note);
        emit frameReady(matToQImage(vis));
    }

    cap.release();
    if (writer.isOpened()) writer.release();

    double secs = timer.elapsed() / 1000.0;
    double speed = secs > 0 ? frame_no / secs : 0.0;
    if (stopRequested_.load()) {
        emit logMsg(tr("[STOPPED] Processed %1 frames (%2 fps)").arg(frame_no).arg(speed, 0, 'f', 1));
    } else {
        emit logMsg(tr("[DONE] %1 frames, max track_id=%2, speed %3 fps, elapsed %4 s, output: %5")
                    .arg(frame_no).arg(max_id).arg(speed, 0, 'f', 1).arg(secs, 0, 'f', 1)
                    .arg(QString::fromStdString(out)));
    }
    emit progress(totalFrames > 0 ? totalFrames : frame_no, totalFrames, QString());
    emit finished(0);
}

// ============================================================
//  主窗口：无边框 + 换肤
// ============================================================
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent, Qt::FramelessWindowHint)   // 去掉系统标题栏与边框
{
    setMinimumSize(860, 640);
    resize(960, 760);

    m_central = new QWidget(this);
    m_central->setAttribute(Qt::WA_StyledBackground, true);
    setCentralWidget(m_central);

    auto* root = new QVBoxLayout(m_central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    // ---- 顶部标题条（可拖拽区域）----
    m_title = new QWidget(m_central);
    m_title->setAttribute(Qt::WA_StyledBackground, true);
    m_title->setFixedHeight(TITLE_H);
    auto* titleRow = new QHBoxLayout(m_title);
    titleRow->setContentsMargins(14, 0, 6, 0);
    titleRow->setSpacing(4);

    m_titleText = new QLabel(tr("DeepTracker Video Tracking"), m_title);
    m_titleText->setStyleSheet("color:#ffffff;font-size:13px;font-weight:600;");
    titleRow->addWidget(m_titleText);
    titleRow->addStretch();

    m_themeBtn = new QPushButton(m_title);
    m_themeBtn->setFixedSize(30, 24);
    m_themeBtn->setIcon(QIcon(QStringLiteral(":/icons/title_theme.png")));
    m_themeBtn->setIconSize(QSize(16, 16));
    m_themeBtn->setToolTip(tr("Theme"));
    m_themeBtn->setStyleSheet(
        "QPushButton{background:transparent;border:none;border-radius:4px;}"
        "QPushButton:hover{background:rgba(255,255,255,0.25);}");
    titleRow->addWidget(m_themeBtn);

    m_minBtn = new QPushButton(m_title);
    m_minBtn->setFixedSize(30, 24);
    m_minBtn->setIcon(titleBarIcon(TitleGlyph::Minimize));
    m_minBtn->setIconSize(QSize(16, 16));
    m_minBtn->setToolTip(tr("Minimize"));
    m_minBtn->setStyleSheet(
        "QPushButton{background:transparent;border:none;border-radius:4px;}"
        "QPushButton:hover{background:rgba(255,255,255,0.25);}");
    connect(m_minBtn, &QPushButton::clicked, this, &QWidget::showMinimized);
    titleRow->addWidget(m_minBtn);

    m_closeBtn = new QPushButton(m_title);
    m_closeBtn->setFixedSize(30, 24);
    m_closeBtn->setIcon(titleBarIcon(TitleGlyph::Close));
    m_closeBtn->setIconSize(QSize(16, 16));
    m_closeBtn->setToolTip(tr("Close"));
    m_closeBtn->setStyleSheet(
        "QPushButton{background:transparent;border:none;border-radius:4px;}"
        "QPushButton:hover{background:#E53935;}"
        "QPushButton:pressed{background:#C62828;}");
    connect(m_closeBtn, &QPushButton::clicked, this, &QWidget::close);
    titleRow->addWidget(m_closeBtn);

    root->addWidget(m_title);

    // ---- 内容区 ----
    root->addWidget(buildUi(), 1);

    // ---- 初始化换肤系统 ----
    initThemes();
    {
        QSettings s(QSettings::IniFormat, QSettings::UserScope,
                    QStringLiteral("DeepTracker"), QStringLiteral("config"));
        int saved = s.value(QStringLiteral("themeIndex"), m_currentTheme).toInt();
        if (saved >= 0 && saved < m_themes.size()) m_currentTheme = saved;
    }
    applyTheme(m_currentTheme);

    connect(m_themeBtn, &QPushButton::clicked, this, &MainWindow::buildThemeMenu);

    // ---- Worker ----
    worker_ = new TrackerWorker(this);
    connect(worker_, &TrackerWorker::logMsg,     this, &MainWindow::appendLog);
    connect(worker_, &TrackerWorker::frameReady, this, &MainWindow::showFrame);
    connect(worker_, &TrackerWorker::progress,   this, &MainWindow::onProgress);
    connect(worker_, &TrackerWorker::finished,   this, &MainWindow::onFinished);
    resolvePaths();
    log(tr("[INFO] Tracking algorithms: https://github.com/shaoshengsong/DeepSORT"));
    log(tr("[INFO] Detection model: %1").arg(params_.detModel.isEmpty() ? tr("(not found)") : params_.detModel));
}

MainWindow::~MainWindow() {
    if (worker_ && worker_->isRunning()) {
        worker_->requestStop();
        worker_->wait(3000);
    }
}

QWidget* MainWindow::buildUi() {
    auto* body = new QWidget(m_central);
    auto* vbox = new QVBoxLayout(body);
    vbox->setContentsMargins(16, 14, 16, 14);
    vbox->setSpacing(12);

    // ---- Input / Output ----
    auto* inBox = new QGroupBox(tr("Input / Output"), body);
    auto* ig = new QGridLayout(inBox);
    ig->setContentsMargins(12, 18, 12, 10);
    ig->setHorizontalSpacing(8);
    ig->setVerticalSpacing(8);
    ig->addWidget(new QLabel(tr("Video:"), inBox), 0, 0);
    edVideo_ = new QLineEdit(inBox); ig->addWidget(edVideo_, 0, 1);
    btnVideo_ = new QPushButton(inBox);
    btnVideo_->setIcon(QIcon(QStringLiteral(":/icons/btn_folder.png")));
    btnVideo_->setIconSize(QSize(16, 16));
    btnVideo_->setToolTip(tr("Browse video..."));
    btnVideo_->setFixedWidth(40);
    ig->addWidget(btnVideo_, 0, 2);
    ig->addWidget(new QLabel(tr("Output:"), inBox), 1, 0);
    edOut_ = new QLineEdit(inBox); ig->addWidget(edOut_, 1, 1);
    btnOut_ = new QPushButton(inBox);
    btnOut_->setIcon(QIcon(QStringLiteral(":/icons/btn_folder.png")));
    btnOut_->setIconSize(QSize(16, 16));
    btnOut_->setToolTip(tr("Browse output..."));
    btnOut_->setFixedWidth(40);
    ig->addWidget(btnOut_, 1, 2);

    // ---- Tracking settings ----
    auto* trBox = new QGroupBox(tr("Tracking Settings"), body);
    auto* tg = new QGridLayout(trBox);
    tg->setContentsMargins(12, 18, 12, 10);
    tg->setHorizontalSpacing(8);
    tg->setVerticalSpacing(8);
    tg->addWidget(new QLabel(tr("Tracker:"), trBox), 0, 0);
    cmbTracker_ = new QComboBox(trBox);
    cmbTracker_->addItem(QStringLiteral("OC-SORT"),        "ocsort");
    cmbTracker_->addItem(QStringLiteral("Deep OC-SORT"),   "deepocsort");
    cmbTracker_->addItem(QStringLiteral("ByteTrack"),      "bytetrack");
    cmbTracker_->addItem(QStringLiteral("DeepSORT"),       "deepsort");
    cmbTracker_->addItem(tr("Detection only"),             "detect");
    tg->addWidget(cmbTracker_, 0, 1);
    tg->addWidget(new QLabel(tr("Confidence:"), trBox), 0, 2);
    edConf_ = new QLineEdit("0.25", trBox); edConf_->setMaximumWidth(72); tg->addWidget(edConf_, 0, 3);
    tg->addWidget(new QLabel(tr("Class:"), trBox), 1, 0);
    edClass_ = new QLineEdit("0", trBox); edClass_->setMaximumWidth(72); tg->addWidget(edClass_, 1, 1);
    tg->addWidget(new QLabel(QStringLiteral("GMC:"), trBox), 1, 2);
    cmbGmc_ = new QComboBox(trBox);
    cmbGmc_->addItems(QStringList() << "none" << "orb" << "sift" << "sparseOptFlow" << "ecc");
    tg->addWidget(cmbGmc_, 1, 3);
    // ReID engages automatically whenever a valid model is found; leave blank to disable
    tg->addWidget(new QLabel(tr("ReID model:"), trBox), 2, 0);
    edReid_ = new QLineEdit(trBox);
    edReid_->setPlaceholderText(tr("Auto-detected; leave blank to disable"));
    tg->addWidget(edReid_, 2, 1, 1, 3);
    btnReid_ = new QPushButton(trBox);
    btnReid_->setIcon(QIcon(QStringLiteral(":/icons/btn_folder.png")));
    btnReid_->setIconSize(QSize(16, 16));
    btnReid_->setToolTip(tr("Browse ReID model..."));
    btnReid_->setFixedWidth(40);
    tg->addWidget(btnReid_, 2, 4);
    tg->addWidget(new QLabel(tr("Note:"), trBox), 3, 0);
    lblReidHint_ = new QLabel(trBox);
    lblReidHint_->setWordWrap(true);
    tg->addWidget(lblReidHint_, 3, 1, 1, 4);

    // ---- Run controls + progress bar ----
    auto* ctrl = new QHBoxLayout;
    btnRun_  = new QPushButton(tr("Start tracking"), body);
    btnRun_->setObjectName("runBtn");
    btnRun_->setIcon(QIcon(QStringLiteral(":/icons/btn_play.png")));
    btnRun_->setIconSize(QSize(16, 16));
    btnRun_->setMinimumWidth(130);
    btnRun_->setCursor(Qt::PointingHandCursor);
    btnStop_ = new QPushButton(tr("Stop"), body);
    btnStop_->setObjectName("stopBtn");
    btnStop_->setIcon(QIcon(QStringLiteral(":/icons/btn_stop.png")));
    btnStop_->setIconSize(QSize(14, 14));
    btnStop_->setMinimumWidth(88);
    btnStop_->setEnabled(false);
    btnStop_->setCursor(Qt::PointingHandCursor);
    prog_ = new QProgressBar(body);
    prog_->setRange(0, 100);
    prog_->setValue(0);
    prog_->setTextVisible(true);
    prog_->setFormat(tr("Idle"));
    ctrl->addWidget(btnRun_);
    ctrl->addWidget(btnStop_);
    ctrl->addSpacing(10);
    ctrl->addWidget(prog_, 1);

    // ---- 预览 ----
    lblPreview_ = new QLabel(body);
    lblPreview_->setObjectName("preview");
    lblPreview_->setAlignment(Qt::AlignCenter);
    lblPreview_->setMinimumHeight(320);
    lblPreview_->setCursor(Qt::PointingHandCursor);

    // ---- 日志 ----
    logEdit_ = new QPlainTextEdit(body);
    logEdit_->setReadOnly(true);
    logEdit_->setMaximumHeight(70);
    logEdit_->setPlaceholderText(tr("Run log"));

    // ---- 底部：核心跟踪算法 GitHub 链接（可点击） ----
    lblAbout_ = new QLabel(body);
    lblAbout_->setTextFormat(Qt::RichText);
    lblAbout_->setOpenExternalLinks(true);   // 点击自动用默认浏览器打开
    lblAbout_->setText(tr(
        "Tracking algorithms: <a href=\"https://github.com/shaoshengsong/DeepSORT\">"
        "https://github.com/shaoshengsong/DeepSORT</a>"));
    lblAbout_->setCursor(Qt::PointingHandCursor);

    vbox->addWidget(inBox);
    vbox->addWidget(trBox);
    vbox->addLayout(ctrl);
    vbox->addWidget(lblPreview_, 1);
    vbox->addWidget(logEdit_);
    vbox->addWidget(lblAbout_);

    connect(btnVideo_, &QPushButton::clicked, this, &MainWindow::onBrowseVideo);
    connect(btnOut_,   &QPushButton::clicked, this, &MainWindow::onBrowseOut);
    connect(btnReid_,  &QPushButton::clicked, this, &MainWindow::onBrowseReid);
    connect(cmbTracker_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onTrackerTypeChanged);
    connect(btnRun_,   &QPushButton::clicked, this, &MainWindow::onRun);
    connect(btnStop_,  &QPushButton::clicked, this, &MainWindow::onStop);

    onTrackerTypeChanged(cmbTracker_->currentIndex());
    return body;
}

// ============================================================
//  换肤系统（移植自界面参考，适配深浅控件配色）
// ============================================================
void MainWindow::initThemes() {
    m_themes.append({ tr("Fresh Green"),  "#22CB64", "#EAF7EF", "#1BA34D", "#5eb161" });
    m_themes.append({ tr("Sky Blue"),     "#2196F3", "#E8F3FD", "#1976D2", "#42a5f5" });
    m_themes.append({ tr("Elegant Purple"),"#9C27B0","#F6EEF8", "#7B1FA2", "#ab47bc" });
    m_themes.append({ tr("Vibrant Orange"),"#FF9800","#FDF3E6", "#F57C00", "#ffa726" });
    m_themes.append({ tr("Passionate Red"),"#F44336","#FDECEB", "#D32F2F", "#ef5350" });
    m_themes.append({ tr("Deep Teal"),    "#009688", "#E7F4F2", "#00796B", "#26a69a" });
}

void MainWindow::applyTheme(int index) {
    if (index < 0 || index >= m_themes.size()) return;
    const Theme& t = m_themes[index];
    m_currentTheme = index;

    // 标题条用深色变体，保证白字清晰
    m_title->setStyleSheet(QString("background-color:%1;").arg(t.darkColor));
    m_titleText->setText(tr("DeepTracker Video Tracking · %1").arg(t.name));

    // 全局控件样式（浅色卡片 + 主题色点缀）
    m_central->setStyleSheet(QString(R"(
        QWidget#centralBody, QWidget { background-color: %1; }
        QGroupBox {
            background-color: #ffffff;
            border: 1px solid #E4E8EC;
            border-radius: 8px;
            margin-top: 22px;
            font-weight: 600;
            color: #37474F;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 12px; top: 4px;
            padding: 0 6px;
            background-color: #ffffff;
            color: %2; font-weight: 600;
        }
        QLabel { color: #455A64; background: transparent; font-weight: 400; }
        QLabel a { color: %2; }
        QCheckBox { color: #455A64; background: transparent; }
        QLineEdit, QComboBox {
            background: #ffffff;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            padding: 4px 8px;
            color: #263238;
        }
        QLineEdit:focus, QComboBox:focus { border: 1px solid %2; }
        QComboBox::drop-down { border: none; width: 22px; }
        QComboBox QAbstractItemView {
            background: #ffffff; color: #263238;
            selection-background-color: %1; selection-color: #ffffff;
            border: 1px solid #CFD8DC;
        }
        QPushButton {
            background: #ffffff;
            border: 1px solid #CFD8DC;
            border-radius: 4px;
            padding: 5px 14px;
            color: #37474F;
        }
        QPushButton:hover { border: 1px solid %2; color: %2; }
        QPushButton:pressed { background: %1; color: #ffffff; }
        QPushButton:disabled { color: #B0BEC5; border: 1px solid #ECEFF1; }
        QPushButton#runBtn {
            background: %2; color: #ffffff; border: none;
            font-weight: 600; padding: 6px 18px; border-radius: 4px;
        }
        QPushButton#runBtn:hover { background: %3; }
        QPushButton#runBtn:pressed { background: %2; }
        QPushButton#runBtn:disabled { background: #B0BEC5; }
        QPushButton#stopBtn {
            background: #ffffff; color: #E53935; border: 1px solid #E53935;
            font-weight: 600; border-radius: 4px;
        }
        QPushButton#stopBtn:hover { background: #FDECEB; }
        QPushButton#stopBtn:disabled { color: #B0BEC5; border: 1px solid #ECEFF1; }
        QProgressBar {
            border: none;
            background: #ECEFF1;
            border-radius: 5px;
            height: 16px;
            text-align: center;
            color: #37474F;
            font-size: 12px;
        }
        QProgressBar::chunk { background: %2; border-radius: 5px; }
        QLabel#preview {
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid #E4E8EC;
        }
        QPlainTextEdit {
            background: #ffffff;
            border: 1px solid #E4E8EC;
            border-radius: 6px;
            color: #37474F;
        }
    )").arg(t.lightColor, t.darkColor, t.accentColor));

    // 持久化主题选择，下次启动自动加载
    QSettings s(QSettings::IniFormat, QSettings::UserScope,
                QStringLiteral("DeepTracker"), QStringLiteral("config"));
    s.setValue(QStringLiteral("themeIndex"), index);
}

void MainWindow::buildThemeMenu() {
    QMenu menu(this);
    for (int i = 0; i < m_themes.size(); ++i) {
        const Theme& t = m_themes[i];
        QAction* action = menu.addAction(t.name);
        action->setCheckable(true);
        action->setChecked(i == m_currentTheme);
        QPixmap pixmap(16, 16);
        pixmap.fill(QColor(t.primaryColor));
        action->setIcon(QIcon(pixmap));
    }
    connect(&menu, &QMenu::triggered, this, &MainWindow::onThemeSelected);
    menu.exec(m_themeBtn->mapToGlobal(QPoint(0, m_themeBtn->height())));
}

void MainWindow::onThemeSelected(QAction* action) {
    if (!action) return;
    for (int i = 0; i < m_themes.size(); ++i) {
        if (m_themes[i].name == action->text()) {
            applyTheme(i);
            break;
        }
    }
}

// ============================================================
//  业务槽函数
// ============================================================
void MainWindow::resolvePaths() {
    QString dir = QCoreApplication::applicationDirPath();
    params_.detModel  = findResource(dir, "models", "yolo26s.onnx");
    params_.reidModel = findResource(dir, "models", "yolo26s-reid.onnx");
    edReid_->setText(params_.reidModel);
}

void MainWindow::onTrackerTypeChanged(int /*idx*/) {
    const QString t = cmbTracker_->currentData().toString();
    bool reidRelevant = (t == "deepocsort" || t == "deepsort");   // trackers that use ReID
    bool gmcNeeded    = (t == "deepocsort");
    bool isDetectOnly = (t == "detect");

    cmbGmc_->setEnabled(gmcNeeded && !isDetectOnly);
    edReid_->setEnabled(reidRelevant && !isDetectOnly);
    btnReid_->setEnabled(reidRelevant && !isDetectOnly);

    QString hint;
    if (t == "ocsort")
        hint = tr("OC-SORT: observation-centric SORT relying only on motion (IoU / velocity direction); no ReID model needed.");
    else if (t == "deepocsort")
        hint = tr("Deep OC-SORT: fuses ReID appearance features automatically when a valid model is present, and supports GMC global motion compensation; degrades to OC-SORT without ReID.");
    else if (t == "bytetrack")
        hint = tr("ByteTrack: two-stage association based on IoU and detection confidence; no ReID model needed.");
    else if (t == "deepsort")
        hint = tr("DeepSORT: cascade matching with minimum cosine distance over a feature gallery and Mahalanobis motion gating; requires a ReID model and falls back to ByteTrack when unavailable.");
    else
        hint = tr("Detection only: no tracker is run; only detection boxes and confidence scores are drawn.");
    lblReidHint_->setText(hint);
}

void MainWindow::onBrowseVideo() {
    QString f = QFileDialog::getOpenFileName(this, tr("Select video"),
        QString(), tr("Video files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All files (*.*)"));
    if (f.isEmpty()) return;
    edVideo_->setText(f);
    if (edOut_->text().isEmpty()) {
        QFileInfo fi(f);
        edOut_->setText(fi.absolutePath() + "/" + fi.completeBaseName() + "_tracked.mp4");
    }
}

void MainWindow::onBrowseOut() {
    QString f = QFileDialog::getSaveFileName(this, tr("Output video"),
        QString(), tr("MP4 (*.mp4);;AVI (*.avi);;All files (*.*)"));
    if (!f.isEmpty()) edOut_->setText(f);
}

void MainWindow::onBrowseReid() {
    QString f = QFileDialog::getOpenFileName(this, tr("Select ReID model"),
        QString(), tr("ONNX (*.onnx);;All files (*.*)"));
    if (!f.isEmpty()) edReid_->setText(f);
}

void MainWindow::onRun() {
    QString video = edVideo_->text();
    if (video.isEmpty()) { log(tr("[ERROR] Please select a video first")); return; }

    params_.video       = video;
    params_.out         = edOut_->text();
    params_.trackerType = cmbTracker_->currentData().toString();
    params_.conf        = edConf_->text().toFloat();
    params_.cls         = edClass_->text().toInt();
    params_.gmcMethod   = cmbGmc_->currentText();
    params_.reidModel   = edReid_->text();
    // Tracker parameters are compiled in (TrackerConfig::defaults_for).

    btnRun_->setEnabled(false);
    btnStop_->setEnabled(true);
    prog_->setRange(0, 100);
    prog_->setValue(0);
    prog_->setFormat(tr("Starting..."));
    log(tr("[INFO] Starting..."));
    worker_->setParams(params_);
    worker_->start();
}

void MainWindow::onStop() {
    if (worker_->isRunning()) {
        worker_->requestStop();
        log(tr("[INFO] Stopping..."));
    }
}

// Real inference progress: current frame / total frames + percentage; busy mode when total is unknown
void MainWindow::onProgress(int frame, int total, const QString& note) {
    if (total > 0) {
        const int pct = qBound(0, frame * 100 / total, 100);
        prog_->setRange(0, total);
        prog_->setValue(qMin(frame, total));
        prog_->setFormat(note.isEmpty()
            ? tr("Frame %1 / %2 (%3%)").arg(frame).arg(total).arg(pct)
            : tr("Frame %1 / %2 (%3%) · %4").arg(frame).arg(total).arg(pct).arg(note));
    } else {
        prog_->setRange(0, 0);
        prog_->setFormat(note.isEmpty()
            ? tr("Processed %1 frames").arg(frame)
            : tr("Processed %1 frames · %2").arg(frame).arg(note));
    }
}

void MainWindow::appendLog(const QString& s) {
    logEdit_->appendPlainText(s);
    logEdit_->verticalScrollBar()->setValue(logEdit_->verticalScrollBar()->maximum());
}

void MainWindow::showFrame(const QImage& img) {
    if (img.isNull()) return;
    QPixmap px = QPixmap::fromImage(img).scaled(lblPreview_->size(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation);
    lblPreview_->setPixmap(px);
}

void MainWindow::onFinished(int code) {
    btnRun_->setEnabled(true);
    btnStop_->setEnabled(false);
    if (code == 0) {
        prog_->setFormat(tr("Done"));
    } else {
        prog_->setRange(0, 100);
        prog_->setFormat(tr("Error"));
    }
}

void MainWindow::log(const QString& s) { appendLog(s); }

void MainWindow::closeEvent(QCloseEvent* e) {
    if (worker_->isRunning()) {
        worker_->requestStop();
        worker_->wait(3000);
    }
    e->accept();
}

// ============================================================
//  无边框窗口拖拽 / 缩放（与界面参考完全一致）
// ============================================================
bool MainWindow::nativeEvent(const QByteArray& eventType, void* message, qintptr* result) {
#ifdef Q_OS_WIN
    Q_UNUSED(eventType)
    MSG* msg = static_cast<MSG*>(message);

    // 拦截 WM_NCHITTEST，手动告诉 Windows 每个像素属于
    // “边框(HT*) / 标题栏(HTCAPTION) / 客户区(其余)”
    if (msg->message == WM_NCHITTEST) {
        const HWND hwnd = reinterpret_cast<HWND>(winId());

        // 物理屏幕坐标 -> 逻辑客户坐标（高 DPI 必须换算）
        RECT wr{};
        GetWindowRect(hwnd, &wr);
        const UINT dpi = GetDpiForWindow(hwnd);
        const double dpr = (dpi == 0) ? 1.0 : dpi / 96.0;
        const int cx = int((GET_X_LPARAM(msg->lParam) - wr.left) / dpr);
        const int cy = int((GET_Y_LPARAM(msg->lParam) - wr.top)  / dpr);

        const int w = width();
        const int h = height();
        const int b = RESIZE_BORDER;

        // 四边/四角 -> 返回 HT*，Windows 自行执行原生缩放
        const bool left   = cx < b;
        const bool right  = cx >= w - b;
        const bool top    = cy < b;
        const bool bottom = cy >= h - b;
        if (left && top)        { *result = HTTOPLEFT;     return true; }
        if (right && top)       { *result = HTTOPRIGHT;    return true; }
        if (right && bottom)    { *result = HTBOTTOMRIGHT; return true; }
        if (left && bottom)     { *result = HTBOTTOMLEFT;  return true; }
        if (left)               { *result = HTLEFT;        return true; }
        if (right)              { *result = HTRIGHT;       return true; }
        if (top)                { *result = HTTOP;         return true; }
        if (bottom)             { *result = HTBOTTOM;      return true; }

        // 顶部标题条（排除「换肤」「最小化」「关闭」按钮）-> HTCAPTION，Windows 原生拖拽
        const int btnLeft = qMin(qMin(m_themeBtn->mapTo(this, QPoint()).x(),
                                      m_minBtn->mapTo(this, QPoint()).x()),
                                 m_closeBtn->mapTo(this, QPoint()).x());
        if (cy < TITLE_H && cx < btnLeft) {
            *result = HTCAPTION;
            return true;
        }
    }
    return false;   // 其余区域交回 Qt 正常处理
#else
    Q_UNUSED(eventType); Q_UNUSED(message); Q_UNUSED(result)
    return false;
#endif
}
