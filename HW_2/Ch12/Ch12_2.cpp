#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

// 將字串轉換為位元向量 (包含 null 終止符)
std::vector<bool> stringToBits (const std::string& s) {
    std::vector<bool> bits;
    for (char c : s) {
        for (int i = 7; i >= 0; --i) {
            bits.push_back((c >> i) & 1);
        }
    }
    char null_terminator = '\0';
    for (int i = 7; i >= 0; --i) {
        bits.push_back((null_terminator >> i) & 1);
    }
    return bits;
}

// 將位元向量轉換回字串 (直到遇到 null 終止符)
std::string bitsToString (const std::vector<bool>& bits) {
    std::string s = "";
    char current_char = 0;
    int bit_count = 0;
    for (bool bit : bits) {
        current_char = (current_char << 1) | bit;
        bit_count++;
        if (bit_count == 8) {
            if (current_char == '\0') {
                break;
            }
            s += current_char;
            current_char = 0;
            bit_count = 0;
        }
    }
    return s;
}

// 比較兩個影像/矩陣是否完全相同 (支持 CV_8U)
bool compareImages (const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) return false;
    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.type() != img2.type()) {
        std::cerr << "Compare Error: Mismatched dimensions or types."
                  << " Img1: " << img1.rows << "x" << img1.cols << " Type:" << img1.type()
                  << " Img2: " << img2.rows << "x" << img2.cols << " Type:" << img2.type() << "\n";
        return false;
    }
    if (img1.type() != CV_8U) {  // This compare assumes CV_8U for simplicity now
        std::cerr << "Compare Warning: compareImages implementation here might be best for CV_8U." << "\n";
    }
    cv::Mat diff;
    cv::compare(img1, img2, diff, cv::CMP_NE);
    return cv::countNonZero(diff) == 0;
}

// --- 空間域: 直方圖平移 (Histogram Shifting) RDH ---

// 計算灰度直方圖 (0-255)
std::map<int, int> calculateHistogram (const cv::Mat& grayImage) {
    CV_Assert(grayImage.type() == CV_8U && grayImage.channels() == 1);
    std::map<int, int> histogram;
    for (int i = 0; i < 256; ++i) {
        histogram[i] = 0;
    }
    for (int r = 0; r < grayImage.rows; ++r) {
        const uchar* rowPtr = grayImage.ptr<uchar>(r);
        for (int c = 0; c < grayImage.cols; ++c) {
            histogram[rowPtr[c]]++;
        }
    }
    return histogram;
}

// 尋找灰度直方圖峰點 (確保 P < 254)
int findPeakBin (const std::map<int, int>& histogram) {
    int peakBin = -1;
    int maxFreq = -1;
    for (int i = 0; i <= 254; ++i) {  // Ensure P <= 254 for simple shifting
        if (histogram.count(i) && histogram.at(i) > maxFreq) {
            maxFreq = histogram.at(i);
            peakBin = i;
        }
    }
    return peakBin;
}

// 空間域 HS 嵌入 (RDH)
// 在原始圖像嵌入訊息，返回含密圖像 (stego image)
cv::Mat embedHistogramShifting (const cv::Mat& originalGrayImage, const std::string& message, int& peakBin) {
    CV_Assert(originalGrayImage.type() == CV_8U && originalGrayImage.channels() == 1);

    std::map<int, int> histogram = calculateHistogram(originalGrayImage);
    peakBin = findPeakBin(histogram);

    if (peakBin == -1) {
        std::cerr << "Error (Spatial HS Embed): Could not find a suitable peak bin (0-254)." << "\n";
        return cv::Mat();
    }
    if (!histogram.count(peakBin) || histogram.at(peakBin) == 0) {
        std::cerr << "Error (Spatial HS Embed): Peak bin " << peakBin << " has zero frequency." << "\n";
        return cv::Mat();
    }
    std::cout << "Spatial HS Embed: Using Peak Bin P = " << peakBin << " (Frequency: " << histogram.at(peakBin) << ")" << "\n";

    std::vector<bool> messageBits = stringToBits(message);
    size_t totalBitsToEmbed = messageBits.size();
    size_t bitsEmbedded = 0;

    if (totalBitsToEmbed > static_cast<size_t>(histogram.at(peakBin))) {
        std::cerr << "Error (Spatial HS Embed): Message too large for image capacity at peak bin " << peakBin
                  << ". Required: " << totalBitsToEmbed << ", Available: " << histogram.at(peakBin) << "\n";
        return cv::Mat();
    }

    cv::Mat stegoImage = originalGrayImage.clone();  // 創建副本進行修改
    int p = peakBin;

    for (int r = 0; r < stegoImage.rows; ++r) {
        uchar* rowPtr = stegoImage.ptr<uchar>(r);
        for (int c = 0; c < stegoImage.cols; ++c) {
            uchar pixelValue = rowPtr[c];  // Read value *before* potential modification in this iteration

            if (pixelValue > p) {
                if (pixelValue == 255) {  // Should not happen if P<=254
                    std::cerr << "Warning (Spatial HS Embed): Pixel value 255 encountered during shift." << "\n";
                } else {
                    rowPtr[c] = pixelValue + 1;  // Shift
                }
            } else if (pixelValue == p) {
                if (bitsEmbedded < totalBitsToEmbed) {
                    if (messageBits[bitsEmbedded]) {  // Embed '1'
                        rowPtr[c] = p + 1;
                    }  // else: Embed '0', remains 'p'
                    bitsEmbedded++;
                }
                // else: message fully embedded, remaining 'p' pixels are unchanged
            }
            // else pixelValue < p : remains unchanged
        }
    }

    std::cout << "Spatial HS Embed: Successfully embedded " << bitsEmbedded << " bits." << "\n";
    if (bitsEmbedded < totalBitsToEmbed) {
        std::cerr << "Warning (Spatial HS Embed): Embedding finished, but not all message bits were embedded." << "\n";
    }

    return stegoImage;  // 返回的是嵌入訊息後的圖像
}

// 空間域 HS 提取與還原 (RDH)
// 從（解密後的）含密圖像中提取訊息，並還原出原始圖像
std::string extractAndRestoreHistogramShifting (const cv::Mat& decryptedStegoImage, int peakBin, cv::Mat& restoredOriginalImage) {
    if (decryptedStegoImage.empty() || decryptedStegoImage.type() != CV_8U || decryptedStegoImage.channels() != 1) {
        std::cerr << "Error (Spatial HS Extract): Input stego image is invalid." << "\n";
        restoredOriginalImage = cv::Mat();
        return "";
    }
    if (peakBin < 0 || peakBin > 254) {
        std::cerr << "Error (Spatial HS Extract): Invalid Peak Bin P = " << peakBin << "\n";
        restoredOriginalImage = cv::Mat();
        return "";
    }

    std::cout << "Spatial HS Extract: Using Peak Bin P = " << peakBin << "\n";

    restoredOriginalImage = decryptedStegoImage.clone();  // 在副本上操作以還原
    std::vector<bool> extractedBits;
    int p = peakBin;

    for (int r = 0; r < restoredOriginalImage.rows; ++r) {
        uchar* rowPtr = restoredOriginalImage.ptr<uchar>(r);
        for (int c = 0; c < restoredOriginalImage.cols; ++c) {
            uchar pixelValue = rowPtr[c];  // Read value before potential restoration

            if (pixelValue == p) {
                extractedBits.push_back(false);  // Extract '0'
                // Value 'p' needs no restoration
            } else if (pixelValue == p + 1) {
                extractedBits.push_back(true);  // Extract '1'
                rowPtr[c] = p;                  // Restore p+1 to p
            } else if (pixelValue > p + 1) {
                // Restore shift
                rowPtr[c] = pixelValue - 1;
            }
            // else pixelValue < p : No message bit, value is original
        }
    }

    std::cout << "Spatial HS Extract: Extracted " << extractedBits.size() << " potential bits." << "\n";
    std::string extractedMessage = bitsToString(extractedBits);
    // restoredOriginalImage 現在應該是還原後的原始圖像了
    return extractedMessage;
}

// --- 簡易流加密/解密 ---

// 使用密鑰生成偽隨機位元組流，並與圖像像素進行 XOR
// key 作為 PRNG 的種子
// 加密和解密使用完全相同的函數和密鑰
cv::Mat encryptDecryptStream (const cv::Mat& inputImage, unsigned int key) {
    CV_Assert(inputImage.type() == CV_8U && inputImage.channels() == 1);
    cv::Mat outputImage = inputImage.clone();
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    // 使用密鑰初始化 Mersenne Twister 引擎
    std::mt19937 rng(key);
    // 設定分佈以生成 0-255 的隨機位元組
    std::uniform_int_distribution<unsigned int> dist(0, 255);  // Use unsigned int for range

    for (int r = 0; r < rows; ++r) {
        uchar* outPtr = outputImage.ptr<uchar>(r);
        const uchar* inPtr = inputImage.ptr<uchar>(r);  // Although we modify outputImage in place, conceptually good to have both
        for (int c = 0; c < cols; ++c) {
            uchar keyByte = static_cast<uchar>(dist(rng));  // 生成隨機位元組
            outPtr[c] = inPtr[c] ^ keyByte;                 // XOR 操作
        }
    }
    return outputImage;
}

int main (void) {
    const std::string imagePath = "../img/image.png";
    const std::string secretMessage = "Hello, World!";
    const unsigned int encryptionKey = 314159;  // 使用一個整數作為加密密鑰 (種子)

    // 讀取原始圖像 (灰度)
    cv::Mat originalImageGray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (originalImageGray.empty()) {
        std::cerr << "Error: Could not read the image as grayscale: " << imagePath << "\n";
        return -1;
    }

    std::cout << "Original image loaded (Grayscale): " << originalImageGray.cols << "x" << originalImageGray.rows << "\n";
    std::cout << "Secret Message: \"" << secretMessage << "\"" << "\n";
    std::cout << "Encryption Key: " << encryptionKey << "\n";
    std::cout << "========================================" << "\n";

    // 使用 HS RDH 將訊息嵌入原始圖像
    std::cout << "Step 1: Embedding message into original image using HS RDH..." << "\n";
    int peakBinUsed = -1;  // 用於儲存 HS 使用的峰點
    cv::Mat stegoImage = embedHistogramShifting(originalImageGray, secretMessage, peakBinUsed);

    if (stegoImage.empty() || peakBinUsed == -1) {
        std::cerr << "HS Embedding failed. Exiting." << "\n";
        return -1;
    }
    std::cout << "HS Embedding successful." << "\n";
    std::cout << "----------------------------------------" << "\n";

    // 加密包含訊息的圖像 (stegoImage)
    std::cout << "Step 2: Encrypting the stego-image..." << "\n";
    cv::Mat encryptedStegoImage = encryptDecryptStream(stegoImage, encryptionKey);
    std::cout << "Encryption successful." << "\n";
    cv::imwrite("ch12_2_encrypted_stego_image.png", encryptedStegoImage);

    // 模擬接收方
    std::cout << "----------------------------------------" << "\n";
    std::cout << "--- Simulating Receiver Side ---" << "\n";

    // 解密收到的圖像
    std::cout << "Step 3: Decrypting the received image..." << "\n";
    // 使用相同的密鑰進行解密
    cv::Mat decryptedStegoImage = encryptDecryptStream(encryptedStegoImage, encryptionKey);
    std::cout << "Decryption successful." << "\n";

    // 驗證解密是否還原了 stegoImage
    if (compareImages(stegoImage, decryptedStegoImage)) {
        std::cout << "Decryption check: Decrypted image matches the intermediate stego-image. (SUCCESS)" << "\n";
    } else {
        std::cout << "Decryption check: Decrypted image DOES NOT match the intermediate stego-image. (FAILED)" << "\n";
        // This indicates a problem in encryption/decryption logic itself.
    }

    // 從解密後的圖像中提取訊息並還原原始圖像
    std::cout << "Step 4: Extracting message and restoring original image using HS RDH..." << "\n";
    cv::Mat restoredOriginalImage;  // 用於存放最終還原的原始圖像
    // 使用一開始嵌入時確定的 peakBin
    std::string extractedMessage = extractAndRestoreHistogramShifting(decryptedStegoImage, peakBinUsed, restoredOriginalImage);

    // 驗證結果
    std::cout << "----------------------------------------" << "\n";
    std::cout << "Step 5: Verifying results..." << "\n";

    // 驗證訊息
    std::cout << "Extracted Message: \"" << extractedMessage << "\"" << "\n";
    if (extractedMessage == secretMessage) {
        std::cout << "Message Verification: SUCCESS" << "\n";
    } else {
        std::cout << "Message Verification: FAILED" << "\n";
    }

    // 驗證圖像還原
    if (!restoredOriginalImage.empty()) {
        if (compareImages(originalImageGray, restoredOriginalImage)) {
            cv::imwrite("ch12_2_restoredOriginalImage.png", restoredOriginalImage);
            std::cout << "Image Restoration Verification: SUCCESS (Original and Restored images are identical)" << "\n";
        } else {
            std::cout << "Image Restoration Verification: FAILED (Original and Restored images differ)" << "\n";
        }
    } else {
        std::cout << "Image Restoration Verification: SKIPPED (Restoration process failed)" << "\n";
    }
    std::cout << "========================================" << "\n";

    return 0;
}