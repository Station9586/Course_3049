#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using cv::Mat;
const int n = 256, m = 256; // image size
int main (void) {
    Mat img(n, m, CV_8U);

    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
        img.at<uchar>(i, j) = (~i & 1)? 255: 0;  // 設定像素值，產生條紋狀影像，奇數行設為 0 (黑色)，偶數行設為 255 (白色)
    }

    cv::imshow("", img);
    cv::waitKey(0);

    vector<uchar> pixels; // 儲存影像的所有像素值

    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
        pixels.emplace_back(img.at<uchar>(i, j));
    }

    vector<pair<uchar, int>> rle; // 儲存 RLE 編碼結果，pair 包含像素值 (uchar) 和重複次數 (int)

    if (!pixels.empty()) {
        uchar last = pixels[0];
        int cnt = 1;
        for (int i = 1; i < pixels.size(); ++i) { // 計算像素值重複次數 (連續相同像素值)
            if (pixels[i] == last) cnt++;
            else {
                rle.emplace_back(last, cnt);
                last = pixels[i];
                cnt = 1;
            }
        }
        rle.emplace_back(last, cnt);
    }

    // cout << "RLE: " << '\n';
    // for (auto &[x, cnt]: rle) {
    //     cout << (int)x << ' ' << cnt << '\n';
    // }

    int original = n * m; // original image size
    int compress = rle.size() * 5; // compressed image size, RLE: uchar + int (1 + 4 bytes)
    double ratio = 1.0 * original / compress;

    cout << "Original (bytes): " << original << '\n';
    cout << "Compress (bytes): " << compress << '\n';
    cout << "Ratio (bytes): " << ratio << "\n";
    return 0;
}