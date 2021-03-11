#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    createWidgets();
    createLayout();
    createConnections();
}

MainWindow::~MainWindow()
{
    cv::destroyAllWindows();
}

void MainWindow::createWidgets()
{
    chooseFeatureComboBox = new QComboBox();
    chooseFeatureComboBox->addItems(QList<QString>() << "Загрузка изображения"
                                                     << "Операции с указателями"
                                                     << "Повышение резкости изображения"
                                                     << "Сглаживание изображения"
                                                     << "Нормализация гистограммы"
                                                     << "Создание панорамы"
                                    );
    applyPushButton = new QPushButton("Запустить");
}


void MainWindow::createLayout()
{
    QWidget *mainWindowWidget = new QWidget();
    QHBoxLayout *mainWindowLayout = new QHBoxLayout();
    mainWindowLayout->addWidget(chooseFeatureComboBox);
    mainWindowLayout->addWidget(applyPushButton);
    mainWindowWidget->setLayout(mainWindowLayout);
    setCentralWidget(mainWindowWidget);
}

void MainWindow::createConnections()
{
    connect(applyPushButton,
            &QPushButton::clicked,
            [&]()
    {
        switch (chooseFeatureComboBox->currentIndex())
        {
            case SYS::toUType(FeatureType::LOADING_IMAGE):
                loadAndShowImage();
                break;
            case SYS::toUType(FeatureType::POINTER_OPERATIONS):
                pixelManipulations();
                break;
            case SYS::toUType(FeatureType::IMAGE_SHARPENING):
                imageSharpening();
                break;
            case SYS::toUType(FeatureType::IMAGE_SMOOTHING):
                imageSmoothing();
                break;
            case SYS::toUType(FeatureType::HISTOGRAM_EQUALIZATION):
                histogramEqualization();
                break;
            default:
                break;
        }
    }
    );
}


void MainWindow::addGaussianNoise(cv::Mat& image, double average, double standard_diviation)
{
    int image_type = (image.channels() == 3) ? CV_16SC3 : CV_16SC1;
    cv::Mat noise_image(image.size(),image_type);
    cv::randn(noise_image,cv::Scalar::all(average),cv::Scalar::all(standard_diviation));
    cv::Mat tmp_img;
    image.convertTo(tmp_img,image_type);
    cv::addWeighted(tmp_img,1.0,noise_image,1.0,0.0,tmp_img);
    tmp_img.convertTo(image,image.type());
}

void MainWindow::loadAndShowImage()
{
    std::string image_path("G:/Stepanov/Projects/ComputerVision/TestingOpenCV/TestingOpenCV/Haze6.jpg");
    cv::Mat image = cv::imread(image_path,cv::IMREAD_COLOR);
    cv::imshow("Sample image",image);
}

void MainWindow::pixelManipulations()
{
    std::string image_path("G:/Stepanov/Projects/ComputerVision/TestingOpenCV/TestingOpenCV/Church.jpg");
    cv::Mat image = cv::imread(image_path,cv::IMREAD_COLOR);
    if (!image.empty())
    {
        cv::imshow("Source image",image);
    }
    else
    {
        return;
    }
    for(int row = 0; row < image.rows; row++)
    {
        uchar* value = image.ptr<uchar>(row);

        for(int j = 0; j < image.cols; j++)
        {
            *value++ = *value ^ 0xFF;//B
            *value++ = *value ^ 0xFF;//G
            *value++ = *value ^ 0xFF;//R
        }
    }
    if(!image.empty())
    {
        cv::imshow("Transformed image",image);
    }
}

void MainWindow::imageSharpening()
{
    cv::Mat img = cv::imread("G:/Stepanov/Projects/ComputerVision/TestingOpenCV/TestingOpenCV/Church.jpg");
    cv::Mat img_with_laplace_sharpening;
    //sharpenWithFilter2D(img,img_with_laplace_sharpening);
    sharpen(img,img_with_laplace_sharpening);
    cv::imshow("Original image is: ", img);
    cv::imshow("Sharpened image is: ", img_with_laplace_sharpening);
}

void MainWindow::sharpen(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert(src.depth() == CV_8U);//смотрим, что тип соответствует unsigned int
    float boost_factor = 1.2;
    cv::Mat laplace_kernel = (cv::Mat_<double>(3,3) << 0,   -1,                0,
                                                       -1,  5 * boost_factor,  -1,
                                                       0,   -1,                0
                              );
    const int n_channels = src.channels();
    dst = cv::Mat::zeros(src.size(),src.type());
    for (int j = 1;j < (src.rows - 1); ++j)
    {
        const uchar* prev_row = src.ptr<uchar>(j - 1);//прошлая строка
        const uchar* cur_row = src.ptr<uchar>(j);//текущая строка
        const uchar* next_row = src.ptr<uchar>(j + 1);//следующая строка
        uchar* output = dst.ptr<uchar>(j);
        for (int i = 0; i < n_channels * (src.cols - 1);++i)
        {
            *output++ = cv::saturate_cast<uchar>(laplace_kernel.at<double>(1,1) * cur_row[i] + laplace_kernel.at<double>(1,0) * cur_row[i - n_channels] + laplace_kernel.at<double>(1,2) * cur_row[i + n_channels] + laplace_kernel.at<double>(0,1) * prev_row[i] + laplace_kernel.at<double>(2,1) * next_row[i]);
        }
    }
    dst.row(0).setTo(cv::Scalar(0));
    dst.row(dst.rows - 1).setTo(cv::Scalar(0));
    dst.col(0).setTo(cv::Scalar(0));
    dst.col(dst.cols - 1).setTo(cv::Scalar(0));
}

void MainWindow::sharpenWithFilter2D(const cv::Mat &src, cv::Mat &dst)
{
    float boost_factor = 1.2;
    cv::Mat laplace_kernel = (cv::Mat_<double>(3,3) << 0, -1, 0, -1, 5 * boost_factor, -1, 0, -1, 0);
    cv::filter2D(src,dst,src.depth(),laplace_kernel);
}

void MainWindow::imageSmoothing()
{
    cv::Mat image = cv::imread("G:/Stepanov/Projects/ComputerVision/TestingOpenCV/TestingOpenCV/Church.jpg");
    cv::Mat noisy_image = image.clone();
    addGaussianNoise(noisy_image,0.,20.);
    cv::imshow("Noisy image",noisy_image);
    cv::Mat box_filter_image_with_kernel_3x3,box_filter_image_with_kernel_5x5,box_filter_image_with_kernel_7x7;
    //сглаживание с использованием единичной матрицы размерности NxN
    cv::boxFilter(noisy_image,box_filter_image_with_kernel_3x3,noisy_image.depth(),cv::Size(3,3));
    cv::imshow("Noisy image with 3x3 ones kernel smoothing",box_filter_image_with_kernel_3x3);
    cv::boxFilter(noisy_image,box_filter_image_with_kernel_5x5,noisy_image.depth(),cv::Size(5,5));
    cv::imshow("Noisy image with 5x5 ones kernel smoothing",box_filter_image_with_kernel_5x5);
    cv::boxFilter(noisy_image,box_filter_image_with_kernel_7x7,noisy_image.depth(),cv::Size(7,7));
    cv::imshow("Noisy image with 7x7 ones kernel smoothing",box_filter_image_with_kernel_7x7);
    //гауссовское сглаживание с ядром NxN
    cv::Mat gaussian_filter_image_with_kernel_3x3,gaussian_filter_image_with_kernel_5x5,gaussian_filter_image_with_kernel_7x7;
    cv::GaussianBlur(noisy_image,gaussian_filter_image_with_kernel_3x3,cv::Size(3,3),20,0);
    cv::imshow("Gaussian blur with 3x3 kernel",gaussian_filter_image_with_kernel_3x3);
    cv::GaussianBlur(noisy_image,gaussian_filter_image_with_kernel_5x5,cv::Size(5,5),20,0);
    cv::imshow("Gaussian blur with 5x5 kernel",gaussian_filter_image_with_kernel_5x5);
    cv::GaussianBlur(noisy_image,gaussian_filter_image_with_kernel_7x7,cv::Size(7,7),20,0);
    cv::imshow("Gaussian blur with 7x7 kernel",gaussian_filter_image_with_kernel_7x7);
    //медианный фильтр с ядром NxN
    cv::Mat median_filter_image_with_kernel_3x3,median_filter_image_with_kernel_5x5,median_filter_image_with_kernel_7x7;
    cv::medianBlur(noisy_image,median_filter_image_with_kernel_3x3,3);
    cv::imshow("Median blur with 3x3 kernel",median_filter_image_with_kernel_3x3);
    cv::medianBlur(noisy_image,median_filter_image_with_kernel_5x5,5);
    cv::imshow("Median blur with 5x5 kernel",median_filter_image_with_kernel_5x5);
    cv::medianBlur(noisy_image,median_filter_image_with_kernel_7x7,7);
    cv::imshow("Median blur with 7x7 kernel",median_filter_image_with_kernel_7x7);
}

void MainWindow::histogramEqualization()
{
    cv::Mat image = cv::imread("G:/Stepanov/Projects/ComputerVision/TestingOpenCV/TestingOpenCV/Church.jpg",cv::IMREAD_GRAYSCALE);
    cv::imshow("Grayscaled image",image);
    int hist_size = 256;
    float range[] = {0, 256};
    const float* hist_range = {range};
    bool uniform = true;
    bool accumulate = false;
    cv::Mat hist;
    cv::calcHist(&image,1,0,cv::Mat(),hist,1,&hist_size,&hist_range,uniform,accumulate);
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / (double)hist_size);
    cv::Mat hist_image(hist_h,hist_w,CV_8UC3,cv::Scalar(255,255,255));
    cv::normalize(hist,hist,0,hist_image.rows,cv::NORM_MINMAX);
    for (int i = 1; i < hist_size;i++)
    {
        cv::line(hist_image,
                 cv::Point(bin_w * (i - 1),hist_h - cvRound(hist.at<float>(i - 1))),
                 cv::Point(bin_w * i,hist_h - cvRound(hist.at<float>(i))),
                 cv::Scalar(255,0,0),1,8,0);
    }
    cv::imshow("Origian histogram",hist_image);
    //проведение глобальной эквализации гистограммы
    cv::Mat out;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->apply(image,out);
    cv::imshow("After clahe",out);
    cv::calcHist(&out,1,0,cv::Mat(),hist,1,&hist_size,&hist_range,uniform,accumulate);
    cv::Mat out_hist_image(hist_h,hist_w,CV_8UC3,cv::Scalar(255,255,255));
    cv::normalize(hist,hist,0,out_hist_image.rows,cv::NORM_MINMAX);
    for (int i = 1; i < hist_size;i++)
    {
        cv::line(out_hist_image,
                 cv::Point(bin_w * (i - 1),hist_h - cvRound(hist.at<float>(i - 1))),
                 cv::Point(bin_w * i,hist_h - cvRound(hist.at<float>(i))),
                 cv::Scalar(255,0,0),1,8,0);
    }
    cv::imshow("CLAHE histogram",out_hist_image);
}




void MainWindow::closeEvent(QCloseEvent *event)
{
    cv::destroyAllWindows();
}
