#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

void processImageXOR (const Mat& input, Mat& output, uchar key) {
    output = input.clone();

    // 遍歷影像的每個像素
    for (int y = 0; y < output.rows; ++y) for (int x = 0; x < output.cols; ++x) {
        // 獲取目前像素的參考 (Vec3b 代表 BGR 三個通道)
        // 使用 reference (&) 可以直接修改像素值，效率較高
        Vec3b& pixel = output.at<Vec3b>(y, x);

        // 對每個顏色通道 (B, G, R) 執行 XOR 操作
        pixel[0] = pixel[0] ^ key;  // Blue 通道
        pixel[1] = pixel[1] ^ key;  // Green 通道
        pixel[2] = pixel[2] ^ key;  // Red 通道
    }
}

int main (void) {
    string imagePath = "../img/image.png";
    Mat originalImage = imread(imagePath, IMREAD_COLOR);  // IMREAD_COLOR 表示以彩色模式載入

    // 同一個金鑰將用於加密和解密
    uchar encryptionKey = 127;

    cout << "使用的 XOR 金鑰: " << static_cast<int>(encryptionKey) << "\n";

    Mat encryptedImage;
    processImageXOR(originalImage, encryptedImage, encryptionKey);

    // 檢查加密過程是否產生有效影像
    if (encryptedImage.empty()) {
        cout << "錯誤：影像加密失敗。\n";
        return -1;
    }

    // 使用相同的金鑰和相同的函式來解密 'encryptedImage'
    Mat decryptedImage;
    processImageXOR(encryptedImage, decryptedImage, encryptionKey);

    // 檢查解密過程是否產生有效影像
    if (decryptedImage.empty()) {
        cout << "錯誤：影像解密失敗。\n";
        return -1;
    }
    // 在視窗中顯示影像
    imshow("Original", originalImage);
    imshow("Encrypted", encryptedImage);
    imshow("Decrypted", decryptedImage);

    // 等待使用者按下任意鍵，否則視窗會立即關閉
    waitKey(0);

    imwrite("ch12_1_encrypted_output.png", encryptedImage);
    imwrite("ch12_1_decrypted_output.png", decryptedImage);

    // 關閉所有 OpenCV 視窗
    destroyAllWindows();

    return 0;
}