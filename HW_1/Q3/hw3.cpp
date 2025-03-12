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


int main (void) {
    cout << fixed << setprecision(4);
    Mat img = cv::imread("../../img/image.png", 0);
    int n = img.rows, m = img.cols;
    if (n & 1) n--; // 確保n, m為偶數
    if (m & 1) m--;

    Mat src = img(cv::Rect(0, 0, m, n)); // 取得偶數大小的原始圖像
    src.convertTo(src, CV_64F); // 轉成 64-bit float

    Mat dst;
    Haar2D(src, dst);  // 執行 2D Haar 轉換

    cout << "Result: " << '\n';
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
        cout << dst.at<double>(i, j) << " \n"[j == m - 1]; // 輸出像素值，並在每行結尾換行
    }
    return 0;
}