#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using cv::Mat;

void Haar2D (const Mat &src, Mat &dst) {
    int n = src.rows, m = src.cols; // n rows, m cols

    Mat tmp = Mat::zeros(src.size(), CV_64F); // 64-bit float，初始化為 0，大小與 src 相同

    for (int i = 0; i < n; ++i) for (int j = 0; j < m; j += 2) { // 針對每一行，每兩個相鄰的 columns 進行處理
        int k = j >> 1; // k 為當前處理的 column / 2
        double a = src.at<double>(i, j), b = src.at<double>(i, j + 1); // 取得相鄰兩個像素值
        tmp.at<double>(i, k) = (a + b) / sqrt(2.0); // approximation coefficient
        tmp.at<double>(i, k + (m >> 1)) = (a - b) / sqrt(2.0); // detail coefficient
    }

    dst = Mat::zeros(src.size(), CV_64F);
    for (int j = 0; j < m; ++j) for (int i = 0; i < n; i += 2) { // 針對每一column，每兩個相鄰的 rows 進行處理
        int k = i >> 1; // 後續同上
        double a = tmp.at<double>(i, j), b = tmp.at<double>(i + 1, j);
        dst.at<double>(k, j) = (a + b) / sqrt(2.0);
        dst.at<double>(k + (n >> 1), j) = (a - b) / sqrt(2.0);
    }
}

void IHaar2D (const Mat &src, Mat &dst) {
    int n = src.rows, m = src.cols; // n rows, m cols

    Mat tmp = Mat::zeros(src.size(), CV_64F); // 64-bit float，初始化為 0，大小與 src 相同

    for (int j = 0; j < m; ++j) for (int i = 0; i < n; i += 2) { // 針對每一column，每兩個相鄰的 rows 進行處理
        int k = i >> 1; // k 為當前處理的 row / 2
        double a = src.at<double>(k, j), b = src.at<double>(k + (n >> 1), j); // 取得相鄰兩個像素值
        tmp.at<double>(i, j) = (a + b) / sqrt(2.0); // approximation coefficient
        tmp.at<double>(i + 1, j) = (a - b) / sqrt(2.0); // detail coefficient
    }

    dst = Mat::zeros(src.size(), CV_64F);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; j += 2) { // 針對每一行，每兩個相鄰的 columns 進行處理
        int k = j >> 1; // 後續同上
        double a = tmp.at<double>(i, k), b = tmp.at<double>(i, k + (m >> 1));
        dst.at<double>(i, j) = (a + b) / sqrt(2.0);
        dst.at<double>(i, j + 1) = (a - b) / sqrt(2.0);
    }
}

int main (void) {
    cout << fixed << setprecision(4);
    Mat img = cv::imread("../../img/image.png", 0);
    cv::imshow("Original", img);
    cv::waitKey(0);
    int n = img.rows, m = img.cols;
    n &= ~1, m &= ~1; // 確保n, m為偶數

    Mat src = img(cv::Rect(0, 0, m, n)); // 取得偶數大小的原始圖像
    src.convertTo(src, CV_64F); // 轉成 64-bit float

    Mat dst;
    Haar2D(src, dst);  // 執行 2D Haar 轉換

    cout << "Result: " << '\n';
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
        cout << dst.at<double>(i, j) << " \n"[j == m - 1]; // 輸出像素值，並在每行結尾換行
    }

    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal);

    Mat display;
    dst.convertTo(display, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::imshow("Haar2D", display);
    cv::waitKey(0);


    Mat ihaar;
    IHaar2D(dst, ihaar);  // 執行 2D Haar 逆轉換

    cv::minMaxLoc(ihaar, &minVal, &maxVal);
    ihaar.convertTo(display, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::imshow("IHaar2D", display);
    cv::waitKey(0);

    cv::destroyAllWindows();
    
    return 0;
}