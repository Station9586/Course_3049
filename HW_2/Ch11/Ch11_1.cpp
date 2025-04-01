#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

// 將字串轉換為位元向量 (包含 null 終止符)
std::vector<bool> stringToBits (const std::string& s) {
    std::vector<bool> bits;
    // 先加入字串本身的字元位元
    for (char c : s) {
        for (int i = 7; i >= 0; --i) {
            bits.push_back((c >> i) & 1);
        }
    }
    // 加入 null 終止符 '\0' 的位元
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
            if (current_char == '\0') { // 遇到 null 終止符
                break;
            }
            s += current_char;
            current_char = 0;
            bit_count = 0;
        }
    }
    return s;
}

// 計算灰度直方圖
std::map<int, int> calculateHistogram (const cv::Mat& grayImage) {
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

// 尋找直方圖峰點 (像素最多的灰度值)
// 返回峰值點 P。確保 P < 254 以避免邊界問題
int findPeakBin (const std::map<int, int>& histogram) {
    int peakBin = -1;
    int maxFreq = -1;
    // 我們需要 P 和 P+1 都有效，所以 P 最大只能是 254
    for (int i = 0; i <= 254; ++i) {
        if (histogram.count(i) && histogram.at(i) > maxFreq) {
            maxFreq = histogram.at(i);
            peakBin = i;
        }
    }
    return peakBin;
}

// 嵌入訊息 (HS RDH)
// 返回：嵌入訊息後的影像
// peakBin: 輸出參數，返回找到的峰值點，提取時需要用到
cv::Mat embedHistogramShifting (const cv::Mat& inputImage, const std::string& message, int& peakBin) {
    // 1. 轉灰度圖
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else if (inputImage.channels() == 1) {
        grayImage = inputImage.clone();
    } else {
        std::cerr << "Error (HS Embed): Unsupported image format (channels=" << inputImage.channels() << ")" << "\n";
        return cv::Mat();
    }

    // 2. 計算直方圖並找峰點 P
    std::map<int, int> histogram = calculateHistogram(grayImage);
    peakBin = findPeakBin(histogram);

    if (peakBin == -1) {
        std::cerr << "Error (HS Embed): Could not find a suitable peak bin (0-254)." << "\n";
        return cv::Mat();
    }
     if (histogram.at(peakBin) == 0) {
         std::cerr << "Error (HS Embed): Peak bin " << peakBin << " has zero frequency. Cannot embed." << "\n";
         return cv::Mat();
     }
    std::cout << "HS Embed: Using Peak Bin P = " << peakBin << " (Frequency: " << histogram.at(peakBin) << ")" << "\n";


    // 3. 準備訊息位元流
    std::vector<bool> messageBits = stringToBits(message);
    size_t totalBitsToEmbed = messageBits.size();
    size_t bitsEmbedded = 0;

    // 檢查容量
    if (totalBitsToEmbed > static_cast<size_t>(histogram.at(peakBin))) {
        std::cerr << "Error (HS Embed): Message too large for image capacity at peak bin " << peakBin
                  << ". Required: " << totalBitsToEmbed << ", Available: " << histogram.at(peakBin) << "\n";
        return cv::Mat();
    }

    // 4. 建立 stego 影像副本並進行嵌入和平移
    cv::Mat stegoImage = grayImage.clone();
    int p = peakBin;

    for (int r = 0; r < stegoImage.rows; ++r) {
        uchar* rowPtr = stegoImage.ptr<uchar>(r);
        for (int c = 0; c < stegoImage.cols; ++c) {
            uchar pixelValue = rowPtr[c];

            if (pixelValue > p) {
                 // 平移: G' = G + 1 if G > P
                 // 需要處理 G=255 的情況，在此簡化實現中 P <= 254 保證了這一點
                rowPtr[c] = pixelValue + 1;
            } else if (pixelValue == p) {
                // 嵌入: G' = P or P+1 based on bit
                if (bitsEmbedded < totalBitsToEmbed) {
                    if (messageBits[bitsEmbedded]) { // bit is 1
                        rowPtr[c] = p + 1;
                    } else { // bit is 0
                        // rowPtr[c] = p; // 保持不變
                    }
                    bitsEmbedded++;
                }
                // 如果訊息嵌入完了，剩下的 P 值像素保持 P 不變
            }
            // else pixelValue < p : 保持不變
        }
    }

    std::cout << "HS Embed: Successfully embedded " << bitsEmbedded << " bits." << "\n";
    if (bitsEmbedded < totalBitsToEmbed) {
         std::cerr << "Warning (HS Embed): Embedding finished, but not all message bits were embedded (Should not happen if capacity check passed)." << "\n";
    }

    return stegoImage;
}

// 提取訊息並還原影像 (HS RDH)
// 返回：提取出的訊息
// restoredImage: 輸出參數，返回還原後的原始影像
std::string extractAndRestoreHistogramShifting (const cv::Mat& stegoImage, int peakBin, cv::Mat& restoredImage) {
    if (stegoImage.empty() || stegoImage.channels() != 1) {
        std::cerr << "Error (HS Extract): Input stego image is invalid (empty or not grayscale)." << "\n";
        restoredImage = cv::Mat();
        return "";
    }
    if (peakBin < 0 || peakBin > 254) {
         std::cerr << "Error (HS Extract): Invalid Peak Bin P = " << peakBin << "\n";
         restoredImage = cv::Mat();
         return "";
    }

    std::cout << "HS Extract: Using Peak Bin P = " << peakBin << "\n";

    restoredImage = stegoImage.clone(); // 開始還原
    std::vector<bool> extractedBits;
    int p = peakBin;

    // 第一次遍歷：提取位元並同時開始還原
    for (int r = 0; r < restoredImage.rows; ++r) {
        uchar* rowPtr = restoredImage.ptr<uchar>(r);
        for (int c = 0; c < restoredImage.cols; ++c) {
            uchar pixelValue = rowPtr[c];

            if (pixelValue == p) {
                extractedBits.push_back(false); // 提取 bit 0
                // 值 P 不需要還原
            } else if (pixelValue == p + 1) {
                extractedBits.push_back(true);  // 提取 bit 1
                rowPtr[c] = p; // 還原 G' = P+1 回 G = P
            } else if (pixelValue > p + 1) {
                // 還原平移: G = G' - 1 if G' > P+1
                rowPtr[c] = pixelValue - 1;
            }
            // else pixelValue < p : 不需要提取也不需要還原
        }
    }

    std::cout << "HS Extract: Extracted " << extractedBits.size() << " potential bits." << "\n";

    // 將提取的位元轉換回字串
    std::string extractedMessage = bitsToString(extractedBits);

    return extractedMessage;
}

// 比較兩個影像是否完全相同
bool compareImages (const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) return false;
    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.type() != img2.type()) {
        return false;
    }
    cv::Mat diff;
    cv::compare(img1, img2, diff, cv::CMP_NE); // diff 中不等於的位置為 255
    return cv::countNonZero(diff) == 0;
}


int main (void) {
    const std::string imagePath = "../img/image.png";
    const std::string secretMessage = "Hello, World!";

    // 讀取原始影像 (灰度)
    cv::Mat originalImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE); // 直接讀取灰度圖

    if (originalImage.empty()) {
        std::cerr << "Error: Could not read the image as grayscale: " << imagePath << "\n";
        return -1;
    }

    std::cout << "Original image loaded (Grayscale): " << originalImage.cols << "x" << originalImage.rows << "\n";
    std::cout << "Secret Message: \"" << secretMessage << "\"" << "\n";
    std::cout << "----------------------------------------" << "\n";

    // --- 直方圖平移 (HS) RDH 處理 ---
    int peakBinUsed = -1; // 用於儲存嵌入時選擇的峰點
    cv::Mat spatialStegoImage = embedHistogramShifting(originalImage, secretMessage, peakBinUsed);

    if (spatialStegoImage.empty() || peakBinUsed == -1) {
         std::cerr << "Spatial HS embedding failed." << "\n";
    } else {
        cv::imwrite("ch11_1_spatial_hs_stego_image.png", spatialStegoImage);

        // 取出訊息並還原影像x
        cv::Mat restoredImage;
        std::string extractedSpatialMessage = extractAndRestoreHistogramShifting(spatialStegoImage, peakBinUsed, restoredImage);

        std::cout << "Extracted Message (Spatial HS): \"" << extractedSpatialMessage << "\"" << "\n";
        // 驗證訊息
        if (extractedSpatialMessage == secretMessage) {
            std::cout << "Spatial HS Message Verification: SUCCESS" << "\n";
        } else {
            std::cout << "Spatial HS Message Verification: FAILED" << "\n";
        }

        // 驗證影像還原
        if (!restoredImage.empty()) {
            cv::imwrite("ch11_1_spatial_hs_restored_image.png", restoredImage);

            if (compareImages(originalImage, restoredImage)) {
                 std::cout << "Spatial HS Image Restoration Verification: SUCCESS (Original and Restored images are identical)" << "\n";
            } else {
                 std::cout << "Spatial HS Image Restoration Verification: FAILED (Original and Restored images differ)" << "\n";
            }
        } else {
             std::cout << "Spatial HS Image Restoration Verification: SKIPPED (Restoration failed)" << "\n";
        }
    }


    return 0;
}