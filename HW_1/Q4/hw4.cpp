#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using cv::Mat;

const int n = 20; // DCT 基底
const double PI = acos(-1);

int main (void) {
    int N = n * n, M = n * n; // 影像的尺寸

    Mat res = Mat::zeros(N, M, CV_64F); // 建立一個 N x M 的 64-bit float 圖像
    
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        Mat tile = Mat::zeros(n, n, CV_64F);  // 建立一個 n x n 的 Mat，用於儲存單個 DCT 基底 tile，資料型態為 64 位元浮點數，並初始化為 0
        double cu = (!i)? sqrt(1.0 / n): sqrt(2.0 / n); // 計算水平方向的縮放係數 cu
        double cv = (!j)? sqrt(1.0 / n): sqrt(2.0 / n); // 計算垂直方向的縮放係數 cv
        for (int x = 0; x < n; ++x) for (int y = 0; y < n; ++y) {  // 計算 DCT 基底值，並儲存到 tile 中
            tile.at<double>(x, y) = cu * cv * cos((x << 1 | 1) * i * PI / (2 * n)) * cos((y << 1 | 1) * j * PI / (2 * n));
        }

        cv::Rect roi(j * n, i * n, n, n);  // 定義 tile 在結果影像 res 中的 ROI 區域，每個 tile 大小為 n x n，依序排列
        tile.copyTo(res(roi)); // 複製算好的結果到 res 中的 ROI 區域
    }

    Mat show;
    cv::normalize(res, show, 0, 255, cv::NORM_MINMAX, CV_8U);  // 正規化結果影像 res，將數值範圍縮放到 0-255，並轉換為 CV_8U 型態
    cv::imshow("DCT", show);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}