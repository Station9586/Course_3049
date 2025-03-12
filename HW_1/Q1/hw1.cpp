#include <opencv2/opencv.hpp>
#include <vector>

using cv::Mat;

int main (void) {
    Mat img = cv::imread("../../img/image.png", 0); // 讀取灰階
    const int n = img.rows, m = img.cols, height = 80; // 圖像大小、標題高度

    std::vector<Mat> v(8); // 8 個位元平面
    for (int k = 0; k < 8; ++k) {
        Mat tmp = Mat::zeros(cv::Size(m, n + height), CV_8UC1);  // 建立黑色圖像
        tmp(cv::Rect(0, 0, m, height)).setTo(cv::Scalar(200)); // 頂部為灰色

        Mat plane = tmp(cv::Rect(0, height, m, n));  // bit 平面區域
        for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
            plane.at<uchar>(i, j) = (uchar)(img.at<uchar>(i, j) & (1 << k)? 255: 0);  // 提取並設定 bit 平面
        }
        // 設定標題 + 圖片
        cv::putText(tmp, "Bit plane " + std::to_string(k + 1), cv::Point(m / 2 - 80, height - 20), 0, 2, cv::Scalar(0), 4, 16);
        v[k] = tmp;
    }

    Mat r1, r2, plane;
    cv::hconcat(std::vector<Mat>(v.begin(), v.begin() + 4), r1); // 上半部水平拼接
    cv::hconcat(std::vector<Mat>(v.begin() + 4, v.end()), r2); // 下半部水平拼接
    cv::vconcat(r1, r2, plane); // 垂直拼接

    cv::imshow("Bitplane", plane);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}