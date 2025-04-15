#include <opencv2/opencv.hpp>  // 主 OpenCV 標頭檔 (通常包含所有需要的模組)
#include <bits/stdc++.h>

// --- LSB 嵌入函數 ---
// 將訊息嵌入到 coverImage 的最低 k 個位元中
cv::Mat embedLSB (const cv::Mat& coverImage, const std::string& message, int k) {
    if (k < 1 || k > 8) {
        throw std::invalid_argument("k 必須介於 1 到 8 之間");
    }

    cv::Mat stegoImage = coverImage.clone();  // 複製一份影像進行修改
    int messageIndex = 0;                     // 目前處理到訊息的第幾個位元
    int messageLen = message.length();

    // 建立清除最低 k 位元的遮罩
    uchar mask = ~((1 << k) - 1);

    for (int r = 0; r < stegoImage.rows; ++r) {
        for (int c = 0; c < stegoImage.cols; ++c) {
            if (messageIndex >= messageLen) {  // 如果訊息已經全部嵌入，就提前結束
                return stegoImage;
            }

            // 取得目前像素值 (灰階影像只有一個通道)
            uchar& pixelValue = stegoImage.at<uchar>(r, c);

            // 1. 清除像素的最低 k 個位元
            uchar clearedPixel = pixelValue & mask;

            // 2. 從訊息中取出接下來的 k 個位元，並組合成一個數值
            uchar messageBitsValue = 0;
            for (int bitPos = 0; bitPos < k; ++bitPos) {
                if (messageIndex < messageLen) {
                    // 從訊息字串取位元 ('0' 或 '1')，轉換成整數 0 或 1
                    int bit = message[messageIndex] - '0';
                    // 將這個位元放到正確的位置 (從高位放到低位)
                    messageBitsValue |= (bit << (k - 1 - bitPos));
                    messageIndex++;
                }
            }

            // 3. 將訊息位元合併到清除後的像素值中
            pixelValue = clearedPixel | messageBitsValue;
        }
    }

    if (messageIndex < messageLen) {
        std::cerr << "警告: 嵌入時影像空間不足以容納完整訊息，只有部分訊息被嵌入。" << "\n";
    }

    return stegoImage;
}

// --- LSB 取出函數 ---
// 從 stegoImage 的最低 k 個位元中取出指定長度的訊息
std::string extractLSB (const cv::Mat& stegoImage, int messageLength, int k) {
    if (k < 1 || k > 8) {
        throw std::invalid_argument("k 必須介於 1 到 8 之間");
    }

    std::string extractedMessage = "";
    int bitsExtracted = 0;

    for (int r = 0; r < stegoImage.rows; ++r) {
        for (int c = 0; c < stegoImage.cols; ++c) {
            if (bitsExtracted >= messageLength) {  // 如果已取出足夠位元，提前結束
                return extractedMessage;
            }

            // 取得目前像素值
            uchar pixelValue = stegoImage.at<uchar>(r, c);

            // 從像素的最低 k 個位元中逐一取出位元
            for (int bitPos = 0; bitPos < k; ++bitPos) {
                if (bitsExtracted < messageLength) {
                    // 提取從高位到低位的第 bitPos 個位元 (在 k 位元區塊內)
                    int bit = (pixelValue >> (k - 1 - bitPos)) & 1;
                    extractedMessage += std::to_string(bit);
                    bitsExtracted++;
                } else {
                    break;  // 訊息長度已達到，跳出內層迴圈
                }
            }
        }
    }
    if (bitsExtracted < messageLength) {
        std::cerr << "警告: 取出訊息時，掃描完整個影像仍未達到指定的訊息長度。" << "\n";
    }

    // 如果訊息長度不是 k 的倍數，最後取出的字串可能會比 messageLength 長一點點
    // 可以選擇截斷到剛好 messageLength
    if (extractedMessage.length() > messageLength) {
        extractedMessage = extractedMessage.substr(0, messageLength);
    }

    return extractedMessage;
}

// --- PSNR 計算函數 ---
// 計算兩個灰階影像之間的峰值信噪比 (dB)
double calculatePSNR (const cv::Mat& img1, const cv::Mat& img2) {
    // 檢查輸入影像是否有效且匹配
    if (img1.empty() || img2.empty()) {
        std::cerr << "錯誤 (PSNR): 輸入影像不可為空。" << "\n";
        return 0.0;
    }
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "錯誤 (PSNR): 輸入影像的尺寸或類型不匹配。" << "\n";
        return 0.0;
    }
    if (img1.type() != CV_8UC1) {
        std::cerr << "警告 (PSNR): 目前僅支援 CV_8UC1 (8位元灰階) 影像。" << "\n";
        // 可以加入對其他類型的支援，但 LSB 通常用於 8 位元
    }

    // 計算均方誤差 (MSE)
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);  // 計算差的絕對值 |img1 - img2|
    diff.convertTo(diff, CV_64F);   // 轉換為 64 位元浮點數以避免平方後溢位
    diff = diff.mul(diff);          // 計算差的平方 (element-wise)

    cv::Scalar sumScalar = cv::sum(diff);                         // 計算所有平方差的總和
    double mse = sumScalar[0] / (double)(img1.rows * img1.cols);  // 除以像素總數得到 MSE

    // 處理 MSE 為 0 的情況 (影像完全相同)
    if (mse <= 1e-10) {                                  // 使用一個小的容忍值來比較浮點數
        return std::numeric_limits<double>::infinity();  // 返回正無窮大
    }

    // 計算 PSNR
    double maxPixelValue = 255.0;  // 對於 8 位元影像
    double psnr = 10.0 * log10((maxPixelValue * maxPixelValue) / mse);

    return psnr;
}

int main (void) {
    std::cout << std::fixed << std::setprecision(4);
    std::string coverImagePath = "../img/image.png";
    std::string secretMessage = "0010111011110001";  // 16 位元的機密訊息
    int targetRows = 512, targetCols = 512;

    cv::Mat coverImage = cv::imread(coverImagePath, cv::IMREAD_GRAYSCALE);


    // --- 檢查與調整影像尺寸和類型 ---
    if (coverImage.rows != targetRows || coverImage.cols != targetCols) {
        std::cout << "Cover 影像尺寸不是 512x512，將會自動調整大小。" << "\n";
        cv::resize(coverImage, coverImage, cv::Size(targetCols, targetRows));
    }
    if (coverImage.channels() != 1) {
        std::cout << "Cover 影像不是灰階，將會自動轉換。" << "\n";
        cv::cvtColor(coverImage, coverImage, cv::COLOR_BGR2GRAY);  // 假設原始是 BGR
    }
    coverImage.convertTo(coverImage, CV_8UC1);  // 確保是 8-bit 無符號單通道

    std::cout << "Cover 影像載入成功 (" << coverImage.cols << "x" << coverImage.rows << ", " << coverImage.channels() << " 通道)" << "\n";
    std::cout << "機密訊息: " << secretMessage << " (長度: " << secretMessage.length() << " 位元)" << "\n";
    std::cout << "----------------------------------------" << "\n";
    cv::imwrite("Result image/cover_image.png", coverImage);  // 儲存原始影像
    cv::imshow("Cover Image", coverImage);                   // 顯示原始影像
    cv::waitKey(0);                                         // 等待按鍵
    cv::destroyWindow("Cover Image");                       // 關閉原始影像視窗

    // --- 處理不同的 LSB 位元數 (k=1, 2, 3) ---
    for (int k = 1; k <= 3; ++k) {
        // --- 檢查容量 ---
        long long imageCapacity = (long long)coverImage.rows * coverImage.cols * k;
        if (secretMessage.length() > imageCapacity) {
            std::cerr << "錯誤: 機密訊息長度 (" << secretMessage.length()
                      << " 位元) 超過影像使用 k=" << k << " LSB 時的容量 ("
                      << imageCapacity << " 位元)。" << "\n";
            continue;  // 跳過這個 k 值
        }

        // --- 嵌入訊息 ---
        cv::Mat stegoImage = embedLSB(coverImage, secretMessage, k);
        std::string stegoImagePath = "Result image/stego_image_k" + std::to_string(k) + ".png";
        cv::imwrite(stegoImagePath, stegoImage);

        // --- 取出訊息 ---
        std::string extractedMessage = extractLSB(stegoImage, secretMessage.length(), k);
        std::cout << "取出的訊息 (k=" << k << "): " << extractedMessage << "\n";

        // --- 驗證訊息 ---
        if (extractedMessage == secretMessage) {
            std::cout << "驗證成功：取出的訊息與原始訊息相符！" << "\n";
        } else {
            std::cout << "驗證失敗：取出的訊息與原始訊息不符！" << "\n";
        }

        // --- 計算 PSNR ---
        double psnr = calculatePSNR(coverImage, stegoImage);
        if (psnr == std::numeric_limits<double>::infinity()) {
            std::cout << "PSNR (k=" << k << "): Infinity (影像完全相同)" << "\n";
        } else {
            std::cout << "PSNR (k=" << k << "): " << psnr << " dB" << "\n";
        }

        // --- 顯示影像---
        cv::imshow("Cover Image", coverImage);
        cv::imshow("Stego Image (k=" + std::to_string(k) + ")", stegoImage);
        cv::waitKey(0);                                                  // 等待按鍵
        cv::destroyWindow("Stego Image (k=" + std::to_string(k) + ")");  // 關閉當前 Stego 視窗

        std::cout << "----------------------------------------" << "\n";
    }

    cv::destroyAllWindows();  // 關閉所有 OpenCV 視窗

    return 0;
}