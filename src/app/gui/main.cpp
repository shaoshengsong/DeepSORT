#include <QApplication>
#include "mainwindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("DeepTracker GUI");
    MainWindow w;
    w.show();
    return app.exec();
}
