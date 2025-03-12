#include <opencv2/opencv.hpp>
#include <iostream>

using cv::Mat;

int main (void) {
    Mat img = cv::imread("../../img/image2.png", 6); // 讀取彩色
    Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1); // 銳化 kernel

    Mat sharped;
    cv::filter2D(img, sharped, img.depth(), kernel); // 銳化處理

    cv::imshow("Original", img);
    cv::waitKey(0); 
    cv::imshow("Sharpened", sharped);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}