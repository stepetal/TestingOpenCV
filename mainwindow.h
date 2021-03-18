#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QHBoxLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QGroupBox>
#include <QDebug>

#include "utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/fast_math.hpp>


class MainWindow : public QMainWindow
{
    Q_OBJECT
    enum class FeatureType : uint8_t {LOADING_IMAGE,          //загрузка и демонстрация изображения
                                      POINTER_OPERATIONS,     //операции с указателями на элементы матрицы
                                      IMAGE_SHARPENING,       //увеличение резкости изображения
                                      IMAGE_SMOOTHING,        //сглаживание изображения
                                      HISTOGRAM_EQUALIZATION, // построение гистограммы и эквализация гистограммы
                                      EDGE_DETECTION,         //обнаружение границ
                                      CONTOUR_DETECTION       //обнаружение контура
                                     };
    QComboBox *chooseFeatureComboBox;
    QPushButton *applyPushButton;
    QPushButton *openImageFilePushButton;
    QLineEdit *filePathLineEdit;
public:
    MainWindow(QWidget *parent = 0);
    void createLayout();
    void createWidgets();
    void createConnections();

    //тестирование функционала OpenCV
    void addGaussianNoise(cv::Mat& image, double average, double standard_diviation);
    void loadAndShowImage();
    void pixelManipulations();
    void imageSharpening();                                 //увеличение резкости изображения
    void sharpen(const cv::Mat& src,cv::Mat& dst);          //применение маски к изображению (аналог cv::filter2D)
    void sharpenWithFilter2D(const cv::Mat& src,cv::Mat& dst);  //применение маски к изображению (метод cv::filter2D)
    void imageSmoothing();
    void histogramEqualization();
    void edgeDetection();
    void contourDetection();


    ~MainWindow();

    // QWidget interface
protected:
    void closeEvent(QCloseEvent *event);
};

#endif // MAINWINDOW_H
