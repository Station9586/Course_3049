#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using cv::Mat;

int main (void) {
    Mat img = cv::imread("../../img/image.png", 0); // 讀取灰階

    Mat blur;
    cv::GaussianBlur(img, blur, cv::Size(5, 5), 0); // 高斯模糊, kernel size = 5x5

    Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1); // 銳化 kernel
    Mat sharped;
    cv::filter2D(blur, sharped, blur.depth(), kernel);  // 以定義的 kernel 對模糊圖像進行銳化

    cv::imshow("Original", img);
    cv::waitKey(0);
    cv::imshow("Blurred", blur);
    cv::waitKey(0);
    cv::imshow("Sharpened", sharped);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}