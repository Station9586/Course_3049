#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using cv::Mat;


uchar clamp (float val) { // 限制值在 0 到 255 之間
    if (val < 0) return 0;
    if (val > 255) return 255;
    return static_cast<uchar>(val);
}

void floydSteinberg (const Mat &src, Mat &dst) { // Floyd-Steinberg dithering algorithm
    Mat input;
    src.convertTo(input, CV_32F);
    int n = src.rows, m = src.cols;
    dst = Mat::zeros(n, m, CV_8U);

    // 遍歷每個像素
    // 對於每個像素，計算其錯誤並將其分配給相鄰的像素
    // 這裡使用了 Floyd-Steinberg 算法的錯誤擴散
    // 將每個像素的值四捨五入到最接近的顏色
    // 並將錯誤分配給右側和下方的像素
    // 右側像素的錯誤分配比例為 7/16，下方像素的錯誤分配比例為 5/16
    // 左下角像素的錯誤分配比例為 3/16，右下角像素的錯誤分配比例為 1/16
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
        float oldPixel = input.at<float>(i, j);
        float newPixel = (oldPixel > 127)? 255: 0;

        dst.at<uchar>(i, j) = newPixel;
        float quantError = oldPixel - newPixel;

        if (j + 1 < m) input.at<float>(i, j + 1) += quantError * 7.0f / 16.0f;
        if (i + 1 < n) {
            if (j > 0) input.at<float>(i + 1, j - 1) += quantError * 3.0f / 16.0f;
            input.at<float>(i + 1, j) += quantError * 5.0f / 16.0f;
            if (j + 1 < m) input.at<float>(i + 1, j + 1) += quantError * 1.0f / 16.0f;
        }
    }
}

int main (void) {
    Mat image = cv::imread("../img/image.png", cv::IMREAD_GRAYSCALE);

    Mat output;

    floydSteinberg(image, output);
    cv::imshow("Original", image);
    cv::imshow("Dithered", output);
    cv::imwrite("ch09_2_dithered.png", output);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}