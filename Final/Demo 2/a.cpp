#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/quality.hpp>  // 引入 OpenCV 品質評估模組

std::vector<bool> string_to_bitstream(const std::string& s) {
    std::vector<bool> bitstream;
    bitstream.reserve(s.length() * 8);
    for (char c : s) {
        for (int i = 0; i < 8; ++i) {
            bitstream.push_back((c >> i) & 1);
        }
    }
    return bitstream;
}

std::string bitstream_to_string(const std::vector<bool>& bitstream) {
    std::string s;
    s.reserve(bitstream.size() / 8);
    for (size_t i = 0; i + 7 < bitstream.size(); i += 8) {
        char c = 0;
        for (size_t j = 0; j < 8; ++j) {
            if (bitstream[i + j]) {
                c |= (1 << j);
            }
        }
        s += c;
    }
    return s;
}

std::vector<bool> int_to_bitstream(int n) {
    const int num_bits = 32;
    std::vector<bool> bitstream(num_bits);
    for (int i = 0; i < num_bits; ++i) {
        bitstream[i] = (n >> i) & 1;
    }
    return bitstream;
}

int bitstream_to_int(const std::vector<bool>& bitstream) {
    const int num_bits = 32;
    int n = 0;
    for (int i = 0; i < num_bits; ++i) {
        if (i < bitstream.size() && bitstream[i]) {
            n |= (1 << i);
        }
    }
    return n;
}

double calculate_block_variance(const cv::Mat& block) {
    if (block.empty() || block.channels() != 1) return -1.0;
    cv::Scalar mean, stddev;
    cv::meanStdDev(block, mean, stddev);
    return stddev[0] * stddev[0];
}

std::string read_secret_message_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// --- 核心演算法：適應性方法 (Adaptive Method) ---

/**
 * @brief (適應性) 執行適應性LSB嵌入 (k=0或k=2)。
 * @param cover_image_const 原始載體影像。
 * @param secret_message 秘密訊息。
 * @param block_size 區塊大小。
 * @param variance_threshold 變異數閾值。
 * @param actual_bits_embedded (回傳用) 實際嵌入的負載位元數。
 * @return 包含隱藏訊息的隱寫影像。
 */
cv::Mat embed_adaptive_lsb(const cv::Mat& cover_image_const, const std::string& secret_message, int block_size, double variance_threshold, long long& actual_bits_embedded) {
    cv::Mat cover_image = cover_image_const.clone();
    if (cover_image.empty() || cover_image.type() != CV_8UC1) {
        return cv::Mat();
    }

    std::vector<bool> message_bitstream = string_to_bitstream(secret_message);
    int message_total_bits = message_bitstream.size();
    const int LENGTH_FIELD_BITS = 32;
    std::vector<bool> length_bitstream = int_to_bitstream(message_total_bits);

    actual_bits_embedded = 0;
    if (cover_image.total() < LENGTH_FIELD_BITS) {
        return cv::Mat();
    }

    // 步驟 1: 將32位元的訊息總長度嵌入到影像最開頭的32個像素中。
    for (int i = 0; i < LENGTH_FIELD_BITS; ++i) {
        uchar& pixel_val = cover_image.at<uchar>(i);
        pixel_val = (pixel_val & 0xFE) | (length_bitstream[i] ? 1 : 0);
    }

    // 步驟 2: 嵌入實際的秘密訊息負載。
    int current_bit_idx = 0;
    const int k_embed_payload = 2;  // 對於複雜區，固定嵌入2個位元。
    long long payload_bits_embedded_count = 0;

    for (int r_block = 0; r_block < cover_image.rows; r_block += block_size) {
        for (int c_block = 0; c_block < cover_image.cols; c_block += block_size) {
            if (current_bit_idx >= message_total_bits) goto embedding_finished_adaptive;

            int current_block_width = std::min(block_size, cover_image.cols - c_block);
            int current_block_height = std::min(block_size, cover_image.rows - r_block);

            // 重要：變異數必須基於 *原始未修改* 的影像計算。
            cv::Mat original_block = cover_image_const(cv::Rect(c_block, r_block, current_block_width, current_block_height));

            if (calculate_block_variance(original_block) < variance_threshold) {
                // 平滑區 (k=0)，不嵌入任何資訊，直接跳過。
                continue;
            }

            // 複雜區 (k=2)，在此區塊的像素中嵌入2個位元。
            for (int br = 0; br < current_block_height; ++br) {
                for (int bc = 0; bc < current_block_width; ++bc) {
                    if ((r_block + br) * cover_image.cols + (c_block + bc) < LENGTH_FIELD_BITS) continue;
                    if (current_bit_idx >= message_total_bits) goto embedding_finished_adaptive;

                    int bits_to_embed_this_pixel = std::min(k_embed_payload, message_total_bits - current_bit_idx);
                    if (bits_to_embed_this_pixel <= 0) goto embedding_finished_adaptive;

                    uchar bits_to_embed_val = 0;
                    for (int k = 0; k < bits_to_embed_this_pixel; ++k)
                        if (message_bitstream[current_bit_idx + k]) bits_to_embed_val |= (1 << k);

                    uchar& pixel_val = cover_image.at<uchar>(r_block + br, c_block + bc);
                    pixel_val = (pixel_val & 0xFC) | (bits_to_embed_val & 0x03);  // 替換最低2個位元。

                    current_bit_idx += bits_to_embed_this_pixel;
                    payload_bits_embedded_count += bits_to_embed_this_pixel;
                }
            }
        }
    }

embedding_finished_adaptive:
    actual_bits_embedded = payload_bits_embedded_count;
    if (current_bit_idx < message_total_bits) std::cout << "警告: 訊息僅部分嵌入 (" << current_bit_idx << "/" << message_total_bits << " bits)。容量不足。\n";
    return cover_image;
}

/**
 * @brief (適應性) 執行適應性LSB提取 (非盲提取)。
 */
std::string extract_adaptive_lsb(const cv::Mat& stego_image, const cv::Mat& cover_image_for_variance, int block_size, double variance_threshold) {
    if (stego_image.empty() || cover_image_for_variance.empty()) return "";

    const int LENGTH_FIELD_BITS = 32;
    if (stego_image.total() < LENGTH_FIELD_BITS) return "";

    // 步驟 1: 從影像最開頭的32個像素提取訊息總長度。
    std::vector<bool> length_bitstream_extracted;
    for (int i = 0; i < LENGTH_FIELD_BITS; ++i)
        length_bitstream_extracted.push_back(stego_image.at<uchar>(i) & 1);
    int message_total_bits_to_extract = bitstream_to_int(length_bitstream_extracted);
    if (message_total_bits_to_extract <= 0) return "";

    // 步驟 2: 提取實際的秘密訊息負載。
    std::vector<bool> extracted_bitstream;
    extracted_bitstream.reserve(message_total_bits_to_extract);
    long long bits_extracted_so_far = 0;
    const int k_extract_payload = 2;  // 預期從複雜區提取2個位元。

    for (int r_block = 0; r_block < stego_image.rows; r_block += block_size) {
        for (int c_block = 0; c_block < stego_image.cols; c_block += block_size) {
            if (bits_extracted_so_far >= message_total_bits_to_extract) goto extraction_finished_adaptive;

            int current_block_width = std::min(block_size, stego_image.cols - c_block);
            int current_block_height = std::min(block_size, stego_image.rows - r_block);

            // 使用原始影像計算變異數，以同步嵌入時的判斷邏輯。
            cv::Mat original_block = cover_image_for_variance(cv::Rect(c_block, r_block, current_block_width, current_block_height));
            if (calculate_block_variance(original_block) < variance_threshold) {
                continue;
            }

            for (int br = 0; br < current_block_height; ++br) {
                for (int bc = 0; bc < current_block_width; ++bc) {
                    if ((r_block + br) * stego_image.cols + (c_block + bc) < LENGTH_FIELD_BITS) continue;
                    if (bits_extracted_so_far >= message_total_bits_to_extract) goto extraction_finished_adaptive;

                    uchar extracted_chunk = stego_image.at<uchar>(r_block + br, c_block + bc) & 0x03;  // 提取最低2個位元。
                    for (int bit_k_idx = 0; bit_k_idx < k_extract_payload; ++bit_k_idx) {
                        if (bits_extracted_so_far < message_total_bits_to_extract) {
                            extracted_bitstream.push_back((extracted_chunk >> bit_k_idx) & 1);
                            bits_extracted_so_far++;
                        }
                    }
                }
            }
        }
    }

extraction_finished_adaptive:
    return bitstream_to_string(extracted_bitstream);
}

// --- 核心演算法：固定方法 (Fixed Method) 作為比較基準 ---

/**
 * @brief (固定) 執行固定LSB嵌入 (k=2)。
 */
cv::Mat embed_fixed_lsb(const cv::Mat& cover_image_const, const std::string& secret_message, long long& actual_bits_embedded) {
    cv::Mat cover_image = cover_image_const.clone();
    if (cover_image.empty() || cover_image.type() != CV_8UC1) {
        return cv::Mat();
    }

    std::vector<bool> message_bitstream = string_to_bitstream(secret_message);
    int message_total_bits = message_bitstream.size();
    const int LENGTH_FIELD_BITS = 32;
    std::vector<bool> length_bitstream = int_to_bitstream(message_total_bits);

    actual_bits_embedded = 0;
    if (cover_image.total() < LENGTH_FIELD_BITS) {
        return cv::Mat();
    }

    // 步驟 1: 與適應性方法相同，先嵌入訊息長度。
    for (int i = 0; i < LENGTH_FIELD_BITS; ++i) {
        uchar& pixel_val = cover_image.at<uchar>(i);
        pixel_val = (pixel_val & 0xFE) | (length_bitstream[i] ? 1 : 0);
    }

    // 步驟 2: 對所有剩餘像素，不分區塊，依序嵌入2個位元。
    int current_bit_idx = 0;
    const int k_embed_payload = 2;
    long long payload_bits_embedded_count = 0;

    for (int i = LENGTH_FIELD_BITS; i < cover_image.total(); ++i) {
        if (current_bit_idx >= message_total_bits) break;

        int bits_to_embed_this_pixel = std::min(k_embed_payload, message_total_bits - current_bit_idx);
        if (bits_to_embed_this_pixel <= 0) break;

        uchar bits_to_embed_val = 0;
        for (int k = 0; k < bits_to_embed_this_pixel; ++k)
            if (message_bitstream[current_bit_idx + k]) bits_to_embed_val |= (1 << k);

        uchar& pixel_val = cover_image.at<uchar>(i);
        pixel_val = (pixel_val & 0xFC) | (bits_to_embed_val & 0x03);

        current_bit_idx += bits_to_embed_this_pixel;
        payload_bits_embedded_count += bits_to_embed_this_pixel;
    }

    actual_bits_embedded = payload_bits_embedded_count;
    if (current_bit_idx < message_total_bits) std::cout << "警告: 訊息僅部分嵌入 (" << current_bit_idx << "/" << message_total_bits << " bits)。容量不足。\n";
    return cover_image;
}

/**
 * @brief (固定) 執行固定LSB提取 (k=2)。
 */
std::string extract_fixed_lsb(const cv::Mat& stego_image) {
    if (stego_image.empty()) return "";

    const int LENGTH_FIELD_BITS = 32;
    if (stego_image.total() < LENGTH_FIELD_BITS) return "";

    // 步驟 1: 提取訊息長度。
    std::vector<bool> length_bitstream_extracted;
    for (int i = 0; i < LENGTH_FIELD_BITS; ++i)
        length_bitstream_extracted.push_back(stego_image.at<uchar>(i) & 1);
    int message_total_bits_to_extract = bitstream_to_int(length_bitstream_extracted);
    if (message_total_bits_to_extract <= 0) return "";

    // 步驟 2: 對所有剩餘像素，依序提取2個位元，直到滿足訊息總長度。
    std::vector<bool> extracted_bitstream;
    extracted_bitstream.reserve(message_total_bits_to_extract);
    long long bits_extracted_so_far = 0;
    const int k_extract_payload = 2;

    for (int i = LENGTH_FIELD_BITS; i < stego_image.total(); ++i) {
        if (bits_extracted_so_far >= message_total_bits_to_extract) break;

        uchar extracted_chunk = stego_image.at<uchar>(i) & 0x03;
        for (int bit_k_idx = 0; bit_k_idx < k_extract_payload; ++bit_k_idx) {
            if (bits_extracted_so_far < message_total_bits_to_extract) {
                extracted_bitstream.push_back((extracted_chunk >> bit_k_idx) & 1);
                bits_extracted_so_far++;
            }
        }
    }
    return bitstream_to_string(extracted_bitstream);
}


int main() {
    const std::string cover_image_path = "img/image4.png";
    const std::string secret_file_path = "secret.txt";
    const std::string stego_fixed_path = "result/stego_fixed_k2.png";
    const std::string stego_adaptive_path = "result/stego_adaptive.png";

    cv::Mat cover_image = cv::imread(cover_image_path, cv::IMREAD_GRAYSCALE);
    if (cover_image.empty()) {
        std::cerr << "錯誤: 無法讀取載體影像 '" << cover_image_path << "'。\n";
        return -1;
    }
    std::string secret_message = read_secret_message_from_file(secret_file_path);
    if (secret_message.empty()) {
        std::cerr << "錯誤: 無法讀取秘密訊息檔案 '" << secret_file_path << "' 或檔案為空。\n";
        return -1;
    }

    std::cout << "=======================================================\n";
    std::cout << "          LSB 隱寫術效能比較分析\n";
    std::cout << "=======================================================\n";
    std::cout << "載體影像: " << cover_image_path << " (" << cover_image.cols << "x" << cover_image.rows << ")\n";
    std::cout << "秘密訊息: " << secret_file_path << " (" << secret_message.length() << " bytes)\n\n";

    // --- 方法一：傳統固定 LSB (k=2) ---
    {
        std::cout << "--- 方法一：傳統固定 LSB (k=2) ---\n";
        long long bits_embedded = 0;
        cv::Mat stego_image = embed_fixed_lsb(cover_image, secret_message, bits_embedded);
        cv::imwrite(stego_fixed_path, stego_image);
        std::cout << "嵌入完成，已儲存至 " << stego_fixed_path << "\n";

    
        double psnr = cv::PSNR(cover_image, stego_image);
        // 使用 OpenCV quality 模組計算 SSIM
        cv::Ptr<cv::quality::QualitySSIM> ssim_calculator = cv::quality::QualitySSIM::create(cover_image);
        cv::Scalar ssim_scalar = ssim_calculator->compute(stego_image);
        double ssim = ssim_scalar[0];  // 對於單通道影像，取第一個值
        double bpp = (double)bits_embedded / (double)cover_image.total();

        std::cout << "效能評估:\n";
        if (psnr > 99)
            std::cout << "  - PSNR: Infinity dB\n";
        else
            std::cout << "  - PSNR: " << psnr << " dB\n";
        std::cout << "  - SSIM: " << ssim << "\n";
        std::cout << "  - 嵌入容量 (Payload): " << bits_embedded << " bits\n";
        std::cout << "  - 嵌入率 (bpp): " << bpp << " bits per pixel\n";

        std::string extracted_message = extract_fixed_lsb(stego_image);
        if (secret_message == extracted_message)
            std::cout << "驗證: 成功!\n\n";
        else
            std::cout << "驗證: 失敗!\n\n";
    }

    // --- 方法二：適應性 LSB (k=0 或 k=2) ---
    {
        const int block_size = 8;
        const double variance_threshold = 30.0;

        std::cout << "--- 方法二：適應性 LSB (k=0, k=2) ---\n";
        std::cout << "參數: block_size=" << block_size << ", variance_threshold=" << variance_threshold << "\n";
        long long bits_embedded = 0;
        cv::Mat stego_image = embed_adaptive_lsb(cover_image, secret_message, block_size, variance_threshold, bits_embedded);
        cv::imwrite(stego_adaptive_path, stego_image);
        std::cout << "嵌入完成，已儲存至 " << stego_adaptive_path << "\n";


        double psnr = cv::PSNR(cover_image, stego_image);
        // 使用 OpenCV quality 模組計算 SSIM
        cv::Ptr<cv::quality::QualitySSIM> ssim_calculator_adaptive = cv::quality::QualitySSIM::create(cover_image);
        cv::Scalar ssim_scalar = ssim_calculator_adaptive->compute(stego_image);
        double ssim = ssim_scalar[0];  // 對於單通道影像，取第一個值
        double bpp = (double)bits_embedded / (double)cover_image.total();

        std::cout << "效能評估:\n";
        if (psnr > 99)
            std::cout << "  - PSNR: Infinity dB\n";
        else
            std::cout << "  - PSNR: " << psnr << " dB\n";
        std::cout << "  - SSIM: " << ssim << "\n";
        std::cout << "  - 嵌入容量 (Payload): " << bits_embedded << " bits\n";
        std::cout << "  - 嵌入率 (bpp): " << bpp << " bits per pixel\n";

        std::string extracted_message = extract_adaptive_lsb(stego_image, cover_image, block_size, variance_threshold);
        if (secret_message == extracted_message)
            std::cout << "驗證: 成功!\n\n";
        else
            std::cout << "驗證: 失敗!\n\n";
    }

    std::cout << "=======================================================\n";

    return 0;
}
