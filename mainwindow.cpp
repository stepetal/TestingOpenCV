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
                                                     << "Обнаружение границ изображения"
                                                     << "Обнаружение контура изображения"
                                                     << "Обнаружение углов при помощи алгоритма Харриса"
                                    );
    applyPushButton = new QPushButton("Запустить");
    openImageFilePushButton = new QPushButton("Открыть");
    filePathLineEdit = new QLineEdit();
    filePathLineEdit->setReadOnly(true);
}


void MainWindow::createLayout()
{
    QWidget *mainWindowWidget = new QWidget();
    QVBoxLayout *mainWindowLayout = new QVBoxLayout();

    QGroupBox *openFileGroupBox = new QGroupBox("Исходное изображение");
    QHBoxLayout *openFileGroupBoxLayout = new QHBoxLayout();
    openFileGroupBoxLayout->addWidget(filePathLineEdit);
    openFileGroupBoxLayout->addWidget(openImageFilePushButton);
    openFileGroupBox->setLayout(openFileGroupBoxLayout);

    QGroupBox *chooseFeaturesGroupBox = new QGroupBox("Выбор операции");
    QVBoxLayout *chooseFeaturesGroupBoxLayout = new QVBoxLayout();
    chooseFeaturesGroupBoxLayout->addWidget(chooseFeatureComboBox);
    chooseFeaturesGroupBox->setLayout(chooseFeaturesGroupBoxLayout);

    QWidget *buttonWidget = new QWidget();
    QHBoxLayout *buttonWidgetLayout = new QHBoxLayout();
    buttonWidgetLayout->addStretch();
    buttonWidgetLayout->addWidget(applyPushButton);
    buttonWidgetLayout->addStretch();
    buttonWidget->setLayout(buttonWidgetLayout);

    mainWindowLayout->addWidget(openFileGroupBox);
    mainWindowLayout->addWidget(chooseFeaturesGroupBox);
    mainWindowLayout->addWidget(buttonWidget);

    mainWindowWidget->setLayout(mainWindowLayout);
    setCentralWidget(mainWindowWidget);
}

void MainWindow::createConnections()
{
    connect(openImageFilePushButton,&QPushButton::clicked,[&]()
    {
        auto imageFilePath = QFileDialog::getOpenFileName(this,"Изображение","./","Image files: (*.jpg *.png)");
        filePathLineEdit->setText(imageFilePath);
    });
    connect(applyPushButton,
            &QPushButton::clicked,
            [&]()
    {
        if (filePathLineEdit->text().isEmpty())
        {
            return;
        }
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
            case SYS::toUType(FeatureType::EDGE_DETECTION):
                edgeDetection();
                break;
            case SYS::toUType(FeatureType::CONTOUR_DETECTION):
                contourDetection();
                break;
            case SYS::toUType(FeatureType::HARRIS_CORNER_DETECTION):
                harrisCornerDetection();
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
    std::string image_path(filePathLineEdit->text().toStdString());
    cv::Mat image = cv::imread(image_path,cv::IMREAD_COLOR);
    if (image.empty())
    {
        return;
    }
    cv::imshow("Sample image",image);
}

void MainWindow::pixelManipulations()
{
    std::string image_path(filePathLineEdit->text().toStdString());
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
    cv::Mat img = cv::imread(filePathLineEdit->text().toStdString());
    if (img.empty())
    {
        return;
    }
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
    cv::Mat image = cv::imread(filePathLineEdit->text().toStdString());
    if (image.empty())
    {
        return;
    }
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
    cv::Mat image = cv::imread(filePathLineEdit->text().toStdString(),cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        return;
    }
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

void MainWindow::edgeDetection()
{
    cv::Mat image = cv::imread(filePathLineEdit->text().toStdString(),cv::IMREAD_COLOR);
    if (image.empty())
    {
        return;
    }
    //определение границ изображения в градациях серого с помощью оператора Собеля
    cv::Mat grayscaled_image,grayscaled_edges;
    cv::medianBlur(image,image,3);
    cv::imshow("Image after median blur",image);
    cv::cvtColor(image,grayscaled_image,cv::COLOR_BGR2GRAY);
    cv::Mat grad_x,grad_y;
    //вычисляем градиент по оси OX
    cv::Sobel(grayscaled_image,grad_x,CV_32F,1,0);
    //находим градиент по оси OY
    cv::Sobel(grayscaled_image,grad_y,CV_32F,0,1);
    cv::pow(grad_x,2,grad_x);
    cv::pow(grad_y,2,grad_y);
    cv::sqrt((grad_x + grad_y),grayscaled_edges);
    grayscaled_edges.convertTo(grayscaled_edges,CV_8U);
    cv::threshold(grayscaled_edges,grayscaled_edges,125,255,cv::THRESH_BINARY);
    cv::imshow("Binarized sobel edges",grayscaled_edges);
    //нахождение границ цветного изображения с помощю оператора Собеля
    std::vector<cv::Mat> channels(image.channels());
    std::vector<cv::Mat> color_edges(image.channels());
    cv::Mat combined_color_edges;
    cv::cvtColor(image,image,cv::COLOR_BGR2HSV);
    cv::split(image,channels);
    for (int i = 0; i < image.channels(); i++)
    {
        cv::Mat color_grad_x,color_grad_y;
        //находим градиент по оси OX
        cv::Sobel(channels[i],color_grad_x,CV_32F,1,0);
        //находим градиент по оси OY
        cv::Sobel(channels[i],color_grad_y,CV_32F,1,0);
        cv::pow(color_grad_x,2,color_grad_x);
        cv::pow(color_grad_y,2,color_grad_y);
        cv::sqrt((color_grad_x + color_grad_y),color_edges[i]);
        //cv::imshow(QString("edge for channel %1").arg(i).toStdString(),color_edges[i]);
    }
    cv::bitwise_or(color_edges[0],color_edges[1],combined_color_edges);
    cv::bitwise_or(combined_color_edges,color_edges[2],combined_color_edges);
    combined_color_edges.convertTo(combined_color_edges,CV_8U);
    cv::imshow("Sobel color edges detection",combined_color_edges);
    //Нахождение гранциц цветного изображения с помощью оператора Лапласа
    cv::Mat combined_laplace_color_edges;
    std::vector<cv::Mat> laplace_color_edges(image.channels());
    for (int i = 0; i < image.channels(); i++)
    {
        cv::Laplacian(image,laplace_color_edges[i],CV_32F);
    }
    cv::bitwise_or(laplace_color_edges[0],laplace_color_edges[1],combined_laplace_color_edges);
    cv::bitwise_or(combined_laplace_color_edges,laplace_color_edges[2],combined_laplace_color_edges);
    combined_laplace_color_edges.convertTo(combined_laplace_color_edges,CV_8U);
    cv::imshow("Laplace color edges detection",combined_laplace_color_edges);
    //Нахождение границ изображения с помощью алгоритма Кэнни
    cv::Mat canny_edges;
    cv::Canny(image,canny_edges,200,250);
    cv::imshow("Canny edge detection",canny_edges);
}

void MainWindow::contourDetection()
{
    cv::Mat image = cv::imread(filePathLineEdit->text().toStdString(),cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        return;
    }
    cv::imshow("Original image",image);
    cv::GaussianBlur(image,image,cv::Size(5,5),0.6,0.6);//сглаживание гауссовским фильтром
    cv::threshold(image,image,127,255,cv::THRESH_BINARY);//обязательное приведение к бинарному виду
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image,contours,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
    cv::Mat canvas = cv::Mat::ones(image.rows,image.cols,CV_8UC3);
    cv::drawContours(canvas,contours,-1,cv::Scalar(0,255,0));
    cv::imshow("Image contours",canvas);
}

void MainWindow::harrisCornerDetection()
{
    //имплементация алгоритма Харриса

    cv::Mat image = cv::imread(filePathLineEdit->text().toStdString(),cv::IMREAD_GRAYSCALE);
    cv::imshow("Original image",image);
    cv::Mat copy_image = image.clone();
    image.convertTo(image,CV_32F);
    float corner_threshold = 0.02;
    cv::Mat grad_x,grad_y;
    cv::Sobel(image.clone(),grad_x,CV_32F,1,0); //находим производную по оси OX
    cv::Sobel(image.clone(),grad_y,CV_32F,0,1); //находим производную по оси OY
    grad_x.convertTo(grad_x,CV_8U);
    cv::imshow("Gradient X",grad_x);
    grad_y.convertTo(grad_y,CV_8U);
    cv::imshow("Gradient Y",grad_y);
    cv::Mat XX = cv::Mat::zeros(cv::Size(image.cols,image.rows),CV_32F);
    cv::Mat XY = cv::Mat::zeros(cv::Size(image.cols,image.rows),CV_32F);
    cv::Mat YY = cv::Mat::zeros(cv::Size(image.cols,image.rows),CV_32F);

    cv::multiply(grad_x.clone(),grad_x.clone(),XX);
    cv::multiply(grad_x.clone(),grad_y.clone(),XY);
    cv::multiply(grad_y.clone(),grad_y.clone(),YY);

    cv::GaussianBlur(XX.clone(),XX,cv::Size(7,7),1,1);
    cv::GaussianBlur(XY.clone(),XY,cv::Size(7,7),1,1);
    cv::GaussianBlur(YY.clone(),YY,cv::Size(7,7),1,1);

    float alpha = 0.05;
    cv::Mat M = cv::Mat::zeros(cv::Size(2,2),CV_32F);
    std::vector<float> eigenvalues(2);
    cv::Mat cornerness = cv::Mat::zeros(cv::Size(image.cols,image.rows),CV_32F);
    for(int i = 0; i < image.rows; ++i)
    {
        float *xx = XX.ptr<float>(i);
        float *xy = XY.ptr<float>(i);
        float *yy = YY.ptr<float>(i);
        for(int j = 0; j < image.cols; ++j)
        {
            M.at<float>(0,0) = xx[j];
            M.at<float>(0,1) = xy[j];
            M.at<float>(1,0) = xy[j];
            M.at<float>(1,1) = yy[j];

            cv::eigen(M,eigenvalues);
            cornerness.at<float>(i,j) = (float)cv::determinant(M) - alpha * ((float)cv::trace(M)[0] * (float)cv::trace(M)[0]);
        }
    }
    cv::threshold(cornerness,cornerness,corner_threshold,255,cv::THRESH_TOZERO);

    // Non-Maximum Suppression
    int rad = 3;

    for (int i = rad; i < image.rows - rad; ++i)
    {
        float *p = cornerness.ptr<float>(i);

        for (int j = rad; j < image.cols - rad; ++j)
        {
            cv::Mat NMS = cornerness(cv::Range(j - rad, j + rad), cv::Range(i - rad, i + rad));
            double minVal, maxVal;
            cv::minMaxLoc(NMS, &minVal, &maxVal);
            if (p[j] < maxVal)
            {
                p[j] = 0;
            }
        }
    }

    for (int i = 0; i < image.rows; ++i)
    {
        float *p = cornerness.ptr<float>(i);
        for (int j = 0; j < image.cols; ++j)
        {
            if (p[j] > 0)
            {
                cv::circle(image, cv::Point(j, i), 1, cv::Scalar(0, 0, 255));
            }
        }
    }

    image.convertTo(image, CV_8U);
    cv::imshow("Harris corner detector", image);
    std::vector<cv::Point2f> corners;
    //реализация алгоритма Харриса на основе GoodFeaturesToTrack
    cv::Mat cornerness_harris = copy_image.clone();
    cv::goodFeaturesToTrack(image,corners,10000,0.01,10,cv::Mat(),3,3,true,0.04);
    foreach (auto corner, corners) {
        cv::circle(cornerness_harris,corner,2,0);
    }
    cv::imshow("Harris corner detector OpenCV",cornerness_harris);

}




void MainWindow::closeEvent(QCloseEvent *event)
{
    cv::destroyAllWindows();
}
