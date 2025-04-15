#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

const int BLOCK_SIZE = 8; // DCT 處理區塊大小
const int COEFF_U = 4;
const int COEFF_V = 1;

// --- DCT 嵌入函數 ---
cv::Mat embedDCT(const cv::Mat& coverImage, const std::string& message, int u, int v) {
    cv::Mat stegoImage = coverImage.clone();
    stegoImage.convertTo(stegoImage, CV_32F); // 轉換為浮點數進行計算

    int messageIndex = 0;
    int messageLen = message.length();
    const float modification_step = 4.0f; // 嘗試更大的步長

    for (int r = 0; r < stegoImage.rows; r += BLOCK_SIZE) {
        for (int c = 0; c < stegoImage.cols; c += BLOCK_SIZE) {
            if (messageIndex >= messageLen) {
                goto end_embedding;
            }

            cv::Mat block = stegoImage(cv::Rect(c, r, BLOCK_SIZE, BLOCK_SIZE));
            cv::dct(block, block); // 進行 DCT

            float& coeff = block.at<float>(u, v); // 取得係數參考
            int coeff_int = static_cast<int>(round(coeff)); // 四捨五入取整
            int current_lsb = std::abs(coeff_int) & 1; // 計算目前奇偶性
            int bit_to_embed = message[messageIndex] - '0'; // 取得要嵌入的位元

            if (current_lsb != bit_to_embed) {
                // 如果奇偶性不符，需要修改係數
                float original_coeff = coeff; // 保留原始浮點值

                // 嘗試直接增減 modification_step 來改變奇偶性
                int parity_after_add = std::abs(static_cast<int>(round(coeff + modification_step))) % 2;
                int parity_after_sub = std::abs(static_cast<int>(round(coeff - modification_step))) % 2;

                if (parity_after_add == bit_to_embed) {
                    coeff += modification_step;
                } else if (parity_after_sub == bit_to_embed) {
                    coeff -= modification_step;
                } else {
                    // *** 強化備用策略 ***
                    // 如果加減 step 都無效，強制設定為一個絕對值較大的整數
                    int new_coeff_int;
                    if (bit_to_embed == 1) { // 需要奇數
                        new_coeff_int = 5;  // 選擇一個奇數
                    } else { // 需要偶數
                        new_coeff_int = -6; // 選擇一個偶數
                    }
                    coeff = static_cast<float>(new_coeff_int);
                }
            }

            cv::idct(block, block); // 進行反 DCT
            messageIndex++;
        }
    }

end_embedding:
    stegoImage.convertTo(stegoImage, CV_8U); // 轉換回 8 位元整數

    if (messageIndex < messageLen) {
        std::cerr << "警告：嵌入過程結束，但只嵌入了 " << messageIndex << " 個位元 (可能有問題)。" << "\n";
    }
    return stegoImage;
}

// --- DCT 取出函數 ---
std::string extractDCT (const cv::Mat& stegoImage, int messageLength, int u, int v) {
    cv::Mat workImage; // 用於處理的影像副本
    // 轉換為浮點數格式以進行 DCT 計算
    stegoImage.convertTo(workImage, CV_32F);

    std::string extractedMessage = ""; // 初始化取出的訊息字串
    int bitsExtracted = 0; // 已取出的位元數量

    // 以 8x8 區塊迭代影像
    for (int r = 0; r < workImage.rows; r += BLOCK_SIZE) {
        for (int c = 0; c < workImage.cols; c += BLOCK_SIZE) {
            if (bitsExtracted >= messageLength) { // 如果已取出足夠長度的訊息，提前結束
                 goto end_extraction;
            }

            // 取得目前的 8x8 區塊 (建立副本以進行 DCT，避免修改 workImage)
            cv::Mat block = workImage(cv::Rect(c, r, BLOCK_SIZE, BLOCK_SIZE)).clone();

            // 進行 DCT 轉換 (in-place)
            cv::dct(block, block);

            // 取得目標係數的值
            float coeff = block.at<float>(u, v);
            // 將係數四捨五入到最接近的整數
            int coeff_int = static_cast<int>(round(coeff));

            // 取出係數整數部分的 LSB (奇偶性) 作為隱藏的位元
            int extracted_bit = std::abs(coeff_int) % 2;

            // 將取出的位元 (0 或 1) 添加到訊息字串中
            extractedMessage += std::to_string(extracted_bit);
            bitsExtracted++; // 增加已取出位元計數
        }
    }

end_extraction: // goto 跳轉標籤

    if (bitsExtracted < messageLength) {
        std::cerr << "警告：取出過程結束，但只取出了 " << bitsExtracted << " 個位元 (預期 " << messageLength << " 個)。" << "\n";
    }

    // 確保返回的訊息字串長度不超過預期長度
    if (extractedMessage.length() > messageLength) {
        return extractedMessage.substr(0, messageLength);
    }

    return extractedMessage; // 返回取出的訊息字串
}


// --- PSNR 計算函數 (與 LSB 範例相同) ---
double calculatePSNR (const cv::Mat& img1, const cv::Mat& img2) {
    // 檢查輸入影像的有效性、尺寸、類型是否匹配且為 CV_8UC1
    if (img1.empty() || img2.empty() || img1.size() != img2.size() || img1.type() != img2.type() || img1.type() != CV_8UC1) {
         std::cerr << "錯誤/警告 (PSNR)：輸入影像無效、不匹配或非 CV_8UC1 格式。" << "\n";
        return 0.0; // 返回 0 或其他錯誤值
    }

    cv::Mat diff;
    cv::absdiff(img1, img2, diff);       // 計算絕對差值 |img1 - img2|
    diff.convertTo(diff, CV_64F);        // 轉換為 64 位元浮點數以避免平方後溢位
    diff = diff.mul(diff);               // 計算差值的平方 (element-wise)

    cv::Scalar sumScalar = cv::sum(diff); // 計算所有平方差的總和
    // 除以像素總數得到均方誤差 (MSE)
    double mse = sumScalar[0] / (double)(img1.rows * img1.cols);

    // 處理 MSE 為 0 (或極小) 的情況，表示影像完全相同
    if (mse <= 1e-10) { // 使用一個小的容忍值比較浮點數
        return std::numeric_limits<double>::infinity(); // 返回正無窮大
    }

    double maxPixelValue = 255.0; // 8 位元影像的最大像素值
    // 計算 PSNR (dB)
    double psnr = 10.0 * log10((maxPixelValue * maxPixelValue) / mse);

    return psnr;
}

// --- 主函數 ---
int main (void) {
    std::cout << std::fixed << std::setprecision(4);
    std::string coverImagePath = "../img/image.png";
    std::string secretMessage = "00101110";        // 要隱藏的 8 位元機密訊息
    int targetRows = 512, targetCols = 512;

    // --- 載入封面影像 ---
    cv::Mat coverImage = cv::imread(coverImagePath, cv::IMREAD_GRAYSCALE); // 以灰階模式載入

    // --- 驗證並準備影像 ---
    if (coverImage.rows != targetRows || coverImage.cols != targetCols) {
        std::cout << "封面影像非 512x512 尺寸，將自動調整大小" << "\n";
        cv::resize(coverImage, coverImage, cv::Size(targetCols, targetRows), 0, 0, cv::INTER_LINEAR);
    }
    
    coverImage.convertTo(coverImage, CV_8UC1); // 確保影像格式為 8 位元無符號單通道

    std::cout << "封面影像載入成功 (" << coverImage.cols << "x" << coverImage.rows << ", " << coverImage.channels() << " 通道)" << "\n";
    std::cout << "機密訊息: " << secretMessage << " (長度: " << secretMessage.length() << " 位元)" << "\n";
    cv::imwrite("Result image/cover_image.png", coverImage); // 儲存封面影像

    // --- 檢查容量 ---
    int numBlocksX = coverImage.cols / BLOCK_SIZE;
    int numBlocksY = coverImage.rows / BLOCK_SIZE;
    int maxCapacity = numBlocksX * numBlocksY; // 每個 8x8 區塊藏 1 位元
    std::cout << "影像容量 (每個 8x8 區塊 1 位元): " << maxCapacity << " 位元" << "\n";

    if (secretMessage.length() > maxCapacity) {
        std::cerr << "錯誤：機密訊息長度超過影像容量。" << "\n";
        return -1;
    }
     if (COEFF_U == 0 && COEFF_V == 0) {
         std::cerr << "錯誤：不能在 DC 係數 (0, 0) 中嵌入資料。" << "\n";
         return -1;
     }
     if (COEFF_U >= BLOCK_SIZE || COEFF_V >= BLOCK_SIZE || COEFF_U < 0 || COEFF_V < 0) {
          std::cerr << "錯誤：選擇的係數索引 (" << COEFF_U << ", " << COEFF_V << ") 超出 8x8 區塊範圍。" << "\n";
         return -1;
     }

    std::cout << "使用 DCT 係數 (" << COEFF_U << ", " << COEFF_V << ") 進行嵌入" << "\n";
    std::cout << "----------------------------------------" << "\n";

    // --- 嵌入訊息 ---
    cv::Mat stegoImage = embedDCT(coverImage, secretMessage, COEFF_U, COEFF_V);
    std::string stegoImagePath = "Result image/stego_image_dct.png";
    cv::imwrite(stegoImagePath, stegoImage);

    // --- 取出訊息 ---
    std::string extractedMessage = extractDCT(stegoImage, secretMessage.length(), COEFF_U, COEFF_V);
    std::cout << "取出的訊息: " << extractedMessage << "\n";

    // --- 驗證訊息 ---
    if (extractedMessage == secretMessage) {
        std::cout << "驗證成功：取出的訊息與原始訊息相符！" << "\n";
    } else {
        std::cout << "驗證失敗：取出的訊息與原始訊息不符！" << "\n";
        std::cout << "原始訊息:  " << secretMessage << "\n";
        std::cout << "取出訊息: " << extractedMessage << "\n";
    }


    double psnr = calculatePSNR(coverImage, stegoImage);
     if (psnr == std::numeric_limits<double>::infinity()) {
          std::cout << "PSNR: Infinity (影像完全相同)" << "\n";
     } else {
          std::cout << "PSNR: " << psnr << " dB" << "\n";
     }
     std::cout << "----------------------------------------" << "\n";


    // --- 顯示影像 ---
    cv::imshow("Cover Image", coverImage);
    cv::imshow("Stego Image - DCT", stegoImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}