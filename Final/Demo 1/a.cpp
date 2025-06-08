#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

// 定義子像素的值
const uchar BLACK_SUBPIXEL = 0;
const uchar WHITE_SUBPIXEL = 255;

// Floyd-Steinberg Dithering 函數
cv::Mat floyd_steinberg_dithering(const cv::Mat& grayscale_image) {
    if (grayscale_image.empty() || grayscale_image.type() != CV_8UC1) {
        std::cerr << "錯誤: 輸入必須是單通道灰階影像 (CV_8UC1)." << std::endl;
        return cv::Mat();
    }
    cv::Mat float_image;
    grayscale_image.convertTo(float_image, CV_32F);
    cv::Mat dithered_image = cv::Mat::zeros(grayscale_image.size(), CV_8UC1);
    for (int r = 0; r < float_image.rows; ++r) {
        for (int c = 0; c < float_image.cols; ++c) {
            float old_pixel = float_image.at<float>(r, c);
            uchar new_pixel = (old_pixel > 127.0f) ? WHITE_SUBPIXEL : BLACK_SUBPIXEL;
            dithered_image.at<uchar>(r, c) = new_pixel;
            float quant_error = old_pixel - static_cast<float>(new_pixel);
            if (c + 1 < float_image.cols) {
                float_image.at<float>(r, c + 1) += quant_error * 7.0f / 16.0f;
            }
            if (r + 1 < float_image.rows) {
                if (c - 1 >= 0) {
                    float_image.at<float>(r + 1, c - 1) += quant_error * 3.0f / 16.0f;
                }
                float_image.at<float>(r + 1, c) += quant_error * 5.0f / 16.0f;
                if (c + 1 < float_image.cols) {
                    float_image.at<float>(r + 1, c + 1) += quant_error * 1.0f / 16.0f;
                }
            }
        }
    }
    return dithered_image;
}

// (2,4)-VC 分享圖產生函數
void generate_shares_2_4(const cv::Mat& binary_secret_image,
                         cv::Mat& share1,
                         cv::Mat& share2,
                         cv::Mat& share3,
                         cv::Mat& share4) {
    if (binary_secret_image.empty() || binary_secret_image.type() != CV_8UC1) {
        std::cerr << "錯誤: 秘密影像必須是單通道二元影像 (CV_8UC1)." << std::endl;
        return;
    }

    int secret_rows = binary_secret_image.rows;
    int secret_cols = binary_secret_image.cols;
    int sub_pixel_dim = 2;  // 2x2 sub-pixels

    share1 = cv::Mat(secret_rows * sub_pixel_dim, secret_cols * sub_pixel_dim, CV_8UC1);
    share2 = cv::Mat(secret_rows * sub_pixel_dim, secret_cols * sub_pixel_dim, CV_8UC1);
    share3 = cv::Mat(secret_rows * sub_pixel_dim, secret_cols * sub_pixel_dim, CV_8UC1);
    share4 = cv::Mat(secret_rows * sub_pixel_dim, secret_cols * sub_pixel_dim, CV_8UC1);

    // 定義基礎的2黑2白2x2模式
    std::vector<cv::Mat> base_2b2w_patterns;
    base_2b2w_patterns.push_back((cv::Mat_<uchar>(2, 2) << BLACK_SUBPIXEL, BLACK_SUBPIXEL, WHITE_SUBPIXEL, WHITE_SUBPIXEL));  // P0
    base_2b2w_patterns.push_back((cv::Mat_<uchar>(2, 2) << WHITE_SUBPIXEL, WHITE_SUBPIXEL, BLACK_SUBPIXEL, BLACK_SUBPIXEL));  // P1 (互補 P0)
    base_2b2w_patterns.push_back((cv::Mat_<uchar>(2, 2) << BLACK_SUBPIXEL, WHITE_SUBPIXEL, BLACK_SUBPIXEL, WHITE_SUBPIXEL));  // P2
    base_2b2w_patterns.push_back((cv::Mat_<uchar>(2, 2) << WHITE_SUBPIXEL, BLACK_SUBPIXEL, WHITE_SUBPIXEL, BLACK_SUBPIXEL));  // P3 (互補 P2)
    base_2b2w_patterns.push_back((cv::Mat_<uchar>(2, 2) << BLACK_SUBPIXEL, WHITE_SUBPIXEL, WHITE_SUBPIXEL, BLACK_SUBPIXEL));  // P4
    base_2b2w_patterns.push_back((cv::Mat_<uchar>(2, 2) << WHITE_SUBPIXEL, BLACK_SUBPIXEL, BLACK_SUBPIXEL, WHITE_SUBPIXEL));  // P5 (互補 P4)

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib_patterns_white(0, base_2b2w_patterns.size() - 1);
    // For black pixels, we use a specific scheme
    // Scheme: S1=A, S2=A, S3=NOT(A), S4=NOT(A) (A and NOT(A) are complementary 2B2W patterns)
    // OR S1=A, S2=NOT(A), S3=A, S4=NOT(A)
    // OR S1=A, S2=NOT(A), S3=NOT(A), S4=A
    // We will use the first scheme: (A, A, NOT(A), NOT(A))
    // We need pairs of complementary patterns. Let's use P0 and P1, P2 and P3, P4 and P5.
    std::uniform_int_distribution<> distrib_complementary_pairs(0, 2);  // 0 for (P0,P1), 1 for (P2,P3), 2 for (P4,P5)

    for (int r = 0; r < secret_rows; ++r) {
        for (int c = 0; c < secret_cols; ++c) {
            uchar secret_pixel_value = binary_secret_image.at<uchar>(r, c);
            cv::Mat p_s1, p_s2, p_s3, p_s4;

            if (secret_pixel_value == WHITE_SUBPIXEL) {
                int choice = distrib_patterns_white(gen);
                p_s1 = base_2b2w_patterns[choice].clone();
                p_s2 = base_2b2w_patterns[choice].clone();
                p_s3 = base_2b2w_patterns[choice].clone();
                p_s4 = base_2b2w_patterns[choice].clone();
            } else {  // BLACK_SUBPIXEL
                int pair_choice = distrib_complementary_pairs(gen);
                cv::Mat pattern_A, pattern_Not_A;

                if (pair_choice == 0) {
                    pattern_A = base_2b2w_patterns[0];      // P0
                    pattern_Not_A = base_2b2w_patterns[1];  // P1
                } else if (pair_choice == 1) {
                    pattern_A = base_2b2w_patterns[2];      // P2
                    pattern_Not_A = base_2b2w_patterns[3];  // P3
                } else {
                    pattern_A = base_2b2w_patterns[4];      // P4
                    pattern_Not_A = base_2b2w_patterns[5];  // P5
                }

                // Randomly assign (A,A,NotA,NotA), (A,NotA,A,NotA), (A,NotA,NotA,A)
                std::uniform_int_distribution<> black_scheme_choice(0, 2);
                int scheme = black_scheme_choice(gen);

                if (scheme == 0) {  // (A,A,NotA,NotA)
                    p_s1 = pattern_A.clone();
                    p_s2 = pattern_A.clone();
                    p_s3 = pattern_Not_A.clone();
                    p_s4 = pattern_Not_A.clone();
                } else if (scheme == 1) {  // (A,NotA,A,NotA)
                    p_s1 = pattern_A.clone();
                    p_s2 = pattern_Not_A.clone();
                    p_s3 = pattern_A.clone();
                    p_s4 = pattern_Not_A.clone();
                } else {  // (A,NotA,NotA,A)
                    p_s1 = pattern_A.clone();
                    p_s2 = pattern_Not_A.clone();
                    p_s3 = pattern_Not_A.clone();
                    p_s4 = pattern_A.clone();
                }
            }

            p_s1.copyTo(share1(cv::Rect(c * sub_pixel_dim, r * sub_pixel_dim, sub_pixel_dim, sub_pixel_dim)));
            p_s2.copyTo(share2(cv::Rect(c * sub_pixel_dim, r * sub_pixel_dim, sub_pixel_dim, sub_pixel_dim)));
            p_s3.copyTo(share3(cv::Rect(c * sub_pixel_dim, r * sub_pixel_dim, sub_pixel_dim, sub_pixel_dim)));
            p_s4.copyTo(share4(cv::Rect(c * sub_pixel_dim, r * sub_pixel_dim, sub_pixel_dim, sub_pixel_dim)));
        }
    }
}

// 疊加任意2張分享圖 (for (2,4)-VC)
cv::Mat overlay_any_2_of_4_shares(const cv::Mat& s_a, const cv::Mat& s_b) {
    if (s_a.empty() || s_b.empty() || s_a.size() != s_b.size() || s_a.type() != CV_8UC1 || s_b.type() != CV_8UC1) {
        std::cerr << "錯誤: 用於疊加的分享圖不正確或大小不匹配." << std::endl;
        return cv::Mat();
    }
    cv::Mat result;
    cv::bitwise_and(s_a, s_b, result);  // AND 疊加
    return result;
}

int main () {
    cv::Mat original_secret_grayscale = cv::imread("img/image.png", cv::IMREAD_GRAYSCALE);
    cv::imwrite("result/original_secret_grayscale.png", original_secret_grayscale);
    if (original_secret_grayscale.empty()) {
        std::cerr << "無法載入灰階秘密影像! 使用內建測試影像。" << std::endl;
        original_secret_grayscale = cv::Mat(50, 100, CV_8UC1);
        // 創建一個中間是黑色方塊，周圍是白色的影像
        cv::rectangle(original_secret_grayscale, cv::Rect(25, 12, 50, 25), cv::Scalar(BLACK_SUBPIXEL), -1);
        cv::rectangle(original_secret_grayscale, cv::Rect(0, 0, 100, 12), cv::Scalar(WHITE_SUBPIXEL), -1);
        cv::rectangle(original_secret_grayscale, cv::Rect(0, 37, 100, 13), cv::Scalar(WHITE_SUBPIXEL), -1);
        cv::rectangle(original_secret_grayscale, cv::Rect(0, 12, 25, 25), cv::Scalar(WHITE_SUBPIXEL), -1);
        cv::rectangle(original_secret_grayscale, cv::Rect(75, 12, 25, 25), cv::Scalar(WHITE_SUBPIXEL), -1);

        cv::imwrite("test_rect_secret.png", original_secret_grayscale);
    }

    cv::Mat dithered_secret_image = floyd_steinberg_dithering(original_secret_grayscale);
    if (dithered_secret_image.empty()) {
        return -1;
    }
    cv::imwrite("result/dithered_secret_2_4.png", dithered_secret_image);
    std::cout << "抖色後的秘密影像已儲存為 dithered_secret_2_4.png" << std::endl;

    cv::Mat s1, s2, s3, s4;
    generate_shares_2_4(dithered_secret_image, s1, s2, s3, s4);

    if (!s1.empty()) {  // 檢查第一張即可，因為它們會一起產生
        cv::imwrite("result/share1_2_4.png", s1);
        cv::imwrite("result/share2_2_4.png", s2);
        cv::imwrite("result/share3_2_4.png", s3);
        cv::imwrite("result/share4_2_4.png", s4);
        std::cout << "基於抖色影像的 (2,4)-VC Shares 1-4 已產生並儲存." << std::endl;

        // 測試疊加不同組合
        cv::Mat revealed_s1s2 = overlay_any_2_of_4_shares(s1, s2);
        cv::Mat revealed_s1s3 = overlay_any_2_of_4_shares(s1, s3);
        cv::Mat revealed_s3s4 = overlay_any_2_of_4_shares(s3, s4);

        if (!revealed_s1s2.empty()) cv::imwrite("result/revealed_s1s2_2_4.png", revealed_s1s2);
        if (!revealed_s1s3.empty()) cv::imwrite("result/revealed_s1s3_2_4.png", revealed_s1s3);
        if (!revealed_s3s4.empty()) cv::imwrite("result/revealed_s3s4_2_4.png", revealed_s3s4);
        std::cout << "疊加部分分享圖的結果已儲存." << std::endl;

        cv::imshow("Original Secret", original_secret_grayscale);
        cv::imshow("Dithered Secret", dithered_secret_image);
        cv::imshow("Revealed S1+S2", revealed_s1s2);
        cv::imshow("Revealed S1+S3", revealed_s1s3);

        cv::waitKey(0);
    }
    return 0;
}