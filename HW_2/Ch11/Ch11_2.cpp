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

// 比較兩個影像/矩陣是否完全相同 (支持 CV_8U 和 CV_32S)
bool compareImages (const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) return false;
    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.type() != img2.type()) {
        std::cerr << "Compare Error: Mismatched dimensions or types."
                  << " Img1: " << img1.rows << "x" << img1.cols << " Type:" << img1.type()
                  << " Img2: " << img2.rows << "x" << img2.cols << " Type:" << img2.type() << "\n";
        return false;
    }
    cv::Mat diff;
    cv::compare(img1, img2, diff, cv::CMP_NE);
    return cv::countNonZero(diff) == 0;
}

// --- 頻率域: IWT-Haar + HS on HH subband RDH ---

// 正向 IWT-Haar (1 level)
cv::Mat forwardIWT_Haar (const cv::Mat& input) {
    CV_Assert(input.type() == CV_8U && input.channels() == 1);
    CV_Assert(input.rows % 2 == 0 && input.cols % 2 == 0);

    cv::Mat src;
    input.convertTo(src, CV_32S);
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat output = cv::Mat::zeros(rows, cols, CV_32SC1);

    // 1. Row transform
    for (int r = 0; r < rows; ++r) {
        cv::Mat rowL = cv::Mat::zeros(1, cols / 2, CV_32SC1);
        cv::Mat rowH = cv::Mat::zeros(1, cols / 2, CV_32SC1);
        for (int c = 0; c < cols / 2; ++c) {
            int x0 = src.at<int>(r, 2 * c);
            int x1 = src.at<int>(r, 2 * c + 1);
            rowH.at<int>(0, c) = x0 - x1;
            rowL.at<int>(0, c) = x1 + static_cast<int>(std::floor(rowH.at<int>(0, c) / 2.0));
        }
        rowL.copyTo(output(cv::Rect(0, r, cols / 2, 1)));
        rowH.copyTo(output(cv::Rect(cols / 2, r, cols / 2, 1)));
    }

    // 2. Column transform
    cv::Mat temp_output = output.clone();
    for (int c = 0; c < cols; ++c) {
        cv::Mat colL = cv::Mat::zeros(rows / 2, 1, CV_32SC1);
        cv::Mat colH = cv::Mat::zeros(rows / 2, 1, CV_32SC1);
        for (int r = 0; r < rows / 2; ++r) {
            int y0 = temp_output.at<int>(2 * r, c);
            int y1 = temp_output.at<int>(2 * r + 1, c);
            colH.at<int>(r, 0) = y0 - y1;
            colL.at<int>(r, 0) = y1 + static_cast<int>(std::floor(colH.at<int>(r, 0) / 2.0));
        }
        colL.copyTo(output(cv::Rect(c, 0, 1, rows / 2)));
        colH.copyTo(output(cv::Rect(c, rows / 2, 1, rows / 2)));
    }

    return output;
}

// 反向 IWT-Haar (1 level)
cv::Mat inverseIWT_Haar (const cv::Mat& input) {
    CV_Assert(input.type() == CV_32SC1);
    CV_Assert(input.rows % 2 == 0 && input.cols % 2 == 0);

    int rows = input.rows;
    int cols = input.cols;
    cv::Mat output = cv::Mat::zeros(rows, cols, CV_32SC1);
    cv::Mat temp_input = input.clone();

    // 1. Column inverse transform
    for (int c = 0; c < cols; ++c) {
        cv::Mat colL = temp_input(cv::Rect(c, 0, 1, rows / 2));
        cv::Mat colH = temp_input(cv::Rect(c, rows / 2, 1, rows / 2));

        for (int r = 0; r < rows / 2; ++r) {
            int l_val = colL.at<int>(r, 0);
            int h_val = colH.at<int>(r, 0);
            int y1 = l_val - static_cast<int>(std::floor(h_val / 2.0));
            int y0 = h_val + y1;

            output.at<int>(2 * r, c) = y0;
            output.at<int>(2 * r + 1, c) = y1;
        }
    }

    // 2. Row inverse transform
    cv::Mat final_output = cv::Mat::zeros(rows, cols, CV_32SC1);
    cv::Mat temp_row_input = output.clone();

    for (int r = 0; r < rows; ++r) {
        cv::Mat rowL = temp_row_input(cv::Rect(0, r, cols / 2, 1));
        cv::Mat rowH = temp_row_input(cv::Rect(cols / 2, r, cols / 2, 1));

        for (int c = 0; c < cols / 2; ++c) {
            int l_val = rowL.at<int>(0, c);
            int h_val = rowH.at<int>(0, c);
            int x1 = l_val - static_cast<int>(std::floor(h_val / 2.0));
            int x0 = h_val + x1;

            final_output.at<int>(r, 2 * c) = x0;
            final_output.at<int>(r, 2 * c + 1) = x1;
        }
    }

    // 使用手動鉗位轉換回 CV_8U
    cv::Mat final_output_8u = cv::Mat(rows, cols, CV_8U);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int val = final_output.at<int>(r, c);
            final_output_8u.at<uchar>(r, c) = cv::saturate_cast<uchar>(val);
        }
    }

    return final_output_8u;
}

// 計算整數矩陣的直方圖
std::map<int, int> calculateHistogramInt (const cv::Mat& intMatrix) {
    CV_Assert(intMatrix.type() == CV_32SC1);
    std::map<int, int> histogram;
    for (int r = 0; r < intMatrix.rows; ++r) {
        const int* rowPtr = intMatrix.ptr<int>(r);
        for (int c = 0; c < intMatrix.cols; ++c) {
            histogram[rowPtr[c]]++;
        }
    }
    return histogram;
}

// 尋找整數直方圖的峰點 (避開可能導致 P+1 溢出的值)
int findPeakBinInt (const std::map<int, int>& histogram) {
    int peakBin = std::numeric_limits<int>::min();
    int maxFreq = -1;
    int int_max = std::numeric_limits<int>::max();

    for (const auto& pair : histogram) {
        int bin = pair.first;
        int freq = pair.second;
        if (freq > maxFreq && bin < int_max) {
            maxFreq = freq;
            peakBin = bin;
        }
    }
    if (maxFreq <= 0) return std::numeric_limits<int>::min();
    return peakBin;
}

// 嵌入訊息到 HH 子帶 using HS
cv::Mat embedHS_IWT_HH (const cv::Mat& iwtCoeffs, const std::string& message, int& peakBinHH) {
    CV_Assert(iwtCoeffs.type() == CV_32SC1);
    int rows = iwtCoeffs.rows;
    int cols = iwtCoeffs.cols;
    int h_rows = rows / 2;
    int h_cols = cols / 2;

    cv::Mat modifiedCoeffs = iwtCoeffs.clone();
    cv::Mat hhBand = modifiedCoeffs(cv::Rect(h_cols, h_rows, h_cols, h_rows));

    std::map<int, int> histogramHH = calculateHistogramInt(hhBand);
    peakBinHH = findPeakBinInt(histogramHH);

    if (peakBinHH == std::numeric_limits<int>::min()) {
        std::cerr << "Error (IWT-HS Embed): Could not find a suitable peak bin in HH subband." << "\n";
        return cv::Mat();
    }
    if (!histogramHH.count(peakBinHH) || histogramHH.at(peakBinHH) == 0) {
        std::cerr << "Error (IWT-HS Embed): Peak bin " << peakBinHH << " in HH has zero frequency or doesn't exist." << "\n";
        return cv::Mat();
    }

    std::cout << "IWT-HS Embed: Using Peak Bin P = " << peakBinHH << " in HH (Frequency: " << histogramHH.at(peakBinHH) << ")" << "\n";

    std::vector<bool> messageBits = stringToBits(message);
    size_t totalBitsToEmbed = messageBits.size();
    size_t bitsEmbedded = 0;

    if (totalBitsToEmbed > static_cast<size_t>(histogramHH.at(peakBinHH))) {
        std::cerr << "Error (IWT-HS Embed): Message too large for HH capacity at peak bin " << peakBinHH
                  << ". Required: " << totalBitsToEmbed << ", Available: " << histogramHH.at(peakBinHH) << "\n";
        return cv::Mat();
    }

    int p = peakBinHH;
    int int_max = std::numeric_limits<int>::max();

    for (int r = 0; r < h_rows; ++r) {
        int* rowPtr = hhBand.ptr<int>(r);
        for (int c = 0; c < h_cols; ++c) {
            int coeffValue = rowPtr[c];

            if (coeffValue > p) {
                if (coeffValue == int_max) {
                    std::cerr << "Warning (IWT-HS Embed): Integer coefficient reached max value during shift. Cannot shift." << "\n";
                } else {
                    rowPtr[c] = coeffValue + 1;
                }
            } else if (coeffValue == p) {
                if (bitsEmbedded < totalBitsToEmbed) {
                    if (messageBits[bitsEmbedded]) {
                        if (p == int_max) {
                            std::cerr << "Warning (IWT-HS Embed): Cannot embed '1' because P is INT_MAX." << "\n";
                        } else {
                            rowPtr[c] = p + 1;
                        }
                    }
                    bitsEmbedded++;
                }
            }
        }
    }

    std::cout << "IWT-HS Embed: Successfully embedded " << bitsEmbedded << " bits into HH subband." << "\n";
    if (bitsEmbedded < totalBitsToEmbed) {
        std::cerr << "Warning (IWT-HS Embed): Embedding finished, but not all message bits were embedded." << "\n";
    }

    return modifiedCoeffs;
}

// 從嵌入訊息的 IWT 係數中提取訊息並還原係數 (Cleaned version)
std::string extractRestoreHS_IWT_HH (const cv::Mat& stegoCoeffs, int peakBinHH, cv::Mat& restoredCoeffs, const cv::Mat& originalHH) {
    CV_Assert(stegoCoeffs.type() == CV_32SC1);
    CV_Assert(originalHH.empty() || originalHH.type() == CV_32SC1);
    int rows = stegoCoeffs.rows;
    int cols = stegoCoeffs.cols;
    int h_rows = rows / 2;
    int h_cols = cols / 2;

    restoredCoeffs = stegoCoeffs.clone();
    cv::Mat hhBand = restoredCoeffs(cv::Rect(h_cols, h_rows, h_cols, h_rows));

    if (peakBinHH == std::numeric_limits<int>::min()) {
        std::cerr << "Error (IWT-HS Extract): Invalid Peak Bin P = " << peakBinHH << "\n";
        restoredCoeffs = cv::Mat();
        return "";
    }
    std::cout << "IWT-HS Extract: Using Peak Bin P = " << peakBinHH << " in HH" << "\n";

    std::vector<bool> extractedBits;
    int p = peakBinHH;

    // 遍歷 HH 子帶：提取位元並還原係數
    for (int r = 0; r < h_rows; ++r) {
        int* rowPtr = hhBand.ptr<int>(r);
        for (int c = 0; c < h_cols; ++c) {
            int coeffValue = rowPtr[c];

            if (coeffValue == p) {
                extractedBits.push_back(false);
                // No change needed
            } else if (coeffValue == p + 1) {
                extractedBits.push_back(true);
                rowPtr[c] = p;  // Apply restoration
            } else if (coeffValue > p + 1) {
                rowPtr[c] = coeffValue - 1;  // Apply restoration
            }
        }
    }

    std::cout << "IWT-HS Extract: Extracted " << extractedBits.size() << " potential bits from HH subband." << "\n";

    // ---> 可選的最終驗證 (Optional Final Check) <---
    cv::Mat currentRestoredHH = restoredCoeffs(cv::Rect(h_cols, h_rows, h_cols, h_rows));
    if (!originalHH.empty() && !currentRestoredHH.empty()) {
        if (compareImages(originalHH, currentRestoredHH)) {
            std::cout << "HH Coefficient Restoration Verification: SUCCESS" << "\n";
        } else {
            std::cout << "HH Coefficient Restoration Verification: FAILED" << "\n";
            cv::Mat diffHH;
            cv::absdiff(originalHH, currentRestoredHH, diffHH);
            int maxDiffInt = 0;
            for (int r = 0; r < diffHH.rows; ++r) {
                const int* ptrDiff = diffHH.ptr<int>(r);
                for (int c = 0; c < diffHH.cols; ++c) {
                    if (ptrDiff[c] > maxDiffInt) {
                        maxDiffInt = ptrDiff[c];
                    }
                }
            }
            std::cout << "Max HH coefficient absolute difference: " << maxDiffInt << "\n";
        }
    }
    // ---> 驗證結束 <---

    std::string extractedMessage = bitsToString(extractedBits);
    return extractedMessage;
}

int main (void) {
    const std::string imagePath = "../img/image.png";
    const std::string secretMessage = "Hello, World!";

    // 讀取原始影像 (灰度)
    cv::Mat originalImageGray = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (originalImageGray.empty()) {
        std::cerr << "Error: Could not read the image as grayscale: " << imagePath << "\n";
        return -1;
    }

    std::cout << "Original image loaded (Grayscale): " << originalImageGray.cols << "x" << originalImageGray.rows << "\n";
    std::cout << "Secret Message: \"" << secretMessage << "\"" << "\n";
    std::cout << "========================================" << "\n";

    // --- 頻率域: IWT-Haar + HS on HH (RDH) 處理 ---
    // 確保頻率域處理的灰度圖尺寸是偶數
    cv::Mat grayFreq = originalImageGray.clone();  // Use a copy
    int rows = grayFreq.rows;
    int cols = grayFreq.cols;
    if (rows % 2 != 0 || cols % 2 != 0) {
        std::cerr << "Warning: Freq domain image dimensions (" << cols << "x" << rows << ") not even. Cropping." << "\n";
        rows = (rows / 2) * 2;
        cols = (cols / 2) * 2;
        if (rows == 0 || cols == 0) {
            std::cerr << "Error: Cropped image dimension is zero." << "\n";
            return -1;  // Or handle differently
        }
        grayFreq = grayFreq(cv::Rect(0, 0, cols, rows)).clone();
        std::cout << "Image cropped to " << cols << "x" << rows << " for Freq Domain" << "\n";
    }

    // 1. 正向 IWT
    cv::Mat iwtResult = forwardIWT_Haar(grayFreq);
    if (iwtResult.empty()) {
        std::cerr << "Forward IWT failed." << "\n";
        return -1;  // Or handle differently
    }

    cv::imwrite("ch11_2_iwt_result.png", iwtResult);
    // 保存原始 HH 以供驗證 
    cv::Mat originalHH = iwtResult(cv::Rect(cols / 2, rows / 2, cols / 2, rows / 2)).clone();

    // 測試 IWT 可逆性
    cv::Mat reconstructed_test = inverseIWT_Haar(iwtResult);
    if (compareImages(grayFreq, reconstructed_test)) {
        std::cout << "IWT Reversibility Test: SUCCESS" << "\n";
    } else {
        std::cout << "IWT Reversibility Test: FAILED" << "\n";
    }

    // 2. 在 IWT 係數 (HH子帶) 中嵌入訊息
    int peakBinHH_used = std::numeric_limits<int>::min();
    cv::Mat stegoCoeffs = embedHS_IWT_HH(iwtResult, secretMessage, peakBinHH_used);

    if (stegoCoeffs.empty() || peakBinHH_used == std::numeric_limits<int>::min()) {
        std::cerr << "IWT-HS embedding failed." << "\n";
    } else {
        // 3. 直接對 stegoCoeffs 提取訊息並還原 IWT 係數
        cv::Mat restoredCoeffs;
        // 傳入 originalHH 以進行可選的內部驗證
        std::string extractedFreqMessage = extractRestoreHS_IWT_HH(stegoCoeffs, peakBinHH_used, restoredCoeffs, originalHH);

        std::cout << "Extracted Message (Freq IWT-HS): \"" << extractedFreqMessage << "\"" << "\n";
        if (extractedFreqMessage == secretMessage) {
            std::cout << "Freq IWT-HS Message Verification: SUCCESS" << "\n";
        } else {
            std::cout << "Freq IWT-HS Message Verification: FAILED" << "\n";
        }

        // 4. 對還原後的係數做反向 IWT 得到還原影像
        if (!restoredCoeffs.empty()) {
            cv::Mat restoredImageFreq = inverseIWT_Haar(restoredCoeffs);
            if (!restoredImageFreq.empty()) {
                // 驗證影像還原
                if (compareImages(grayFreq, restoredImageFreq)) {  // Compare with the potentially cropped grayFreq
                    cv::imwrite("ch11_2_restored_image.png", restoredImageFreq);
                    std::cout << "Freq IWT-HS Image Restoration Verification: SUCCESS" << "\n";
                } else {
                    std::cout << "Freq IWT-HS Image Restoration Verification: FAILED" << "\n";
                    cv::Mat diffImage;
                    cv::absdiff(grayFreq, restoredImageFreq, diffImage);
                    double minValAbs, maxValAbs;
                    cv::minMaxLoc(diffImage, &minValAbs, &maxValAbs);
                    std::cout << "Max pixel absolute difference: " << maxValAbs << "\n";
                }
            } else {
                std::cout << "Freq IWT-HS Image Restoration Verification: SKIPPED (Inverse IWT failed on restored coefficients)" << "\n";
            }
        } else {
            std::cout << "Freq IWT-HS Image Restoration Verification: SKIPPED (Coefficient restoration failed or not attempted)" << "\n";
        }
    }

    std::cout << "========================================" << "\n";

    return 0;
}