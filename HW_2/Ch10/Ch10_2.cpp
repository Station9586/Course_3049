#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using cv::Mat;

// 定義 ZigZag 掃描順序中前幾個 DCT 係數的索引 (用於選擇嵌入位置)
const vector<vector<int>> zigZagID = {
    {0,1}, {1,0}, {2,0}, {1,1}, {0,2}, {0,3}, {1,2}, {2,1}, {3,0}, {4,0}, // 前 10 個
    {3,1}, {2,2}, {1,3}, {0,4}, {0,5}, {1,4}, {2,3}, {3,2}, {4,1}, {5,0}  // 第 11 到 20 個
};

// 將字串轉換成位元 vector (同上一個範例)
vector<bool> stobits(const string& s) {
    vector<bool> bits;
    bits.reserve(s.length() << 3);

    for (char c : s) for (int i = 7; ~i; --i) {
        bits.push_back((c >> i) & 1);
    }

    return bits;
}

// 將位元 vector 轉換回字串 (同上一個範例)
string bitstos(const vector<bool>& bits) {
    string s = "";
    char now = 0;
    int cnt = 0;

    s.reserve(bits.size() >> 3);

    for (bool b : bits) {
        now = (now << 1) | b;
        ++cnt;

        if (cnt == 8) {
            if (now == '\0') break;
            s += now;
            now = 0;
            cnt = 0;
        }
    }

    return s;
}

// 將訊息位元嵌入到 8x8 DCT 區塊的係數中
// src: 8x8 的 DCT 係數區塊 (CV_32F, 會被修改)
// bits: 完整的訊息位元 vector
// id: 當前處理到訊息位元的索引 (傳參考，會被修改)
// 返回值: 在此區塊中成功嵌入的位元數量
int embedDCT (Mat &src, const vector<bool> &bits, int &id) {
    int cnt = 0; // 記錄在此區塊嵌入的位元數
    int used = 10; // 指定要使用 ZigZag 順序中的前幾個係數來嵌入

    // 遍歷指定的 ZigZag 係數索引
    for (int i = 0; i < used and id < bits.size(); ++i) {
        // 確保 zigZagID 索引有效
        if (i >= zigZagID.size()) break;
        int r = zigZagID[i][0], c = zigZagID[i][1]; // 取得係數的行(r)和列(c)

        // 確保係數座標在 8x8 範圍內
        if (r < 0 || r >= 8 || c < 0 || c >= 8) continue;

        float coef_f = src.at<float>(r, c); // 取得浮點數形式的 DCT 係數
        // 四捨五入到最接近的整數，用於修改 LSB
        int coef_i = static_cast<int>(round(coef_f));

        // 修改整數係數的最低有效位 (LSB)
        // coef_i & ~1: 將 LSB 清零 (AND 1111...1110)
        // | bits[id++]: 將訊息位元設定到 LSB (OR 0000...000x)，並將 id 移至下一位元
        int new_coef_i = (coef_i & ~1) | bits[id++];

        // 將修改後的整數係數轉回浮點數並寫回 DCT 區塊
        src.at<float>(r, c) = static_cast<float>(new_coef_i);
        ++cnt; // 增加嵌入計數
    }
    return cnt; // 回傳在此區塊嵌入的位元數
}

// 從 8x8 DCT 區塊的係數中提取 LSB
// src: 8x8 的 DCT 係數區塊 (CV_32F, const 不會被修改)
// 返回值: 從此區塊提取出的位元 vector
vector<bool> extractDCT (const Mat &src) {
    vector<bool> bits; // 存放提取出的位元
    int used = 10; // 指定要從 ZigZag 順序中的前幾個係數提取
    bits.reserve(used); // 預留空間

    // 遍歷指定的 ZigZag 係數索引
    for (int i = 0; i < used; ++i) {
        // 確保 zigZagID 索引有效
        if (i >= zigZagID.size()) break;
        int r = zigZagID[i][0], c = zigZagID[i][1]; // 取得係數的行(r)和列(c)

        // 確保係數座標在 8x8 範圍內
        if (r < 0 || r >= 8 || c < 0 || c >= 8) continue;

        float coef_f = src.at<float>(r, c); // 取得浮點數形式的 DCT 係數
        // 四捨五入到最接近的整數
        int coef_i = static_cast<int>(round(coef_f));

        // 提取 LSB (整數值 AND 1) 並加入結果 vector
        bits.push_back(coef_i & 1);
    }
    return bits; // 回傳提取出的位元
}

int main (void) {
    string inputImagePath = "../img/image.png";
    string stegoImagePath = "ch10_2_dct_stego_output.png";

    Mat img = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    string msg = "Hello, World!";

    msg = msg + '\0'; // 添加結束符號
    int originalRows = img.rows; // 保留原始尺寸用於裁剪
    int originalCols = img.cols;

    // 計算補邊後的維度 n, m (向上取整到 8 的倍數)
    int n = ((originalRows + 7) & ~7);
    int m = ((originalCols + 7) & ~7);

    // 補邊
    Mat Padded;
    cv::copyMakeBorder(img, Padded, 0, n - originalRows, 0, m - originalCols, cv::BORDER_REPLICATE);

    // 轉換為浮點數
    Mat floatImg;
    Padded.convertTo(floatImg, CV_32F);

    // 準備訊息位元
    vector<bool> bits = stobits(msg);
    int N = bits.size(); // 使用 N 儲存總位元數 (含 null)

    // 檢查容量
    int totalBlocks = (n >> 3) * (m >> 3);
    int mx = totalBlocks * 10; // 估計容量


    // 執行嵌入
    int now = 0; // 當前處理到的位元索引 (使用你的原始變數名)
    Mat stego = floatImg.clone(); // 在副本上操作

    for (int i = 0; i < n && now < N; i += 8) { // 使用維度 n
        for (int j = 0; j < m && now < N; j += 8) { // 使用維度 m
            Mat block = stego(cv::Rect(j, i, 8, 8));
            Mat dctBlock;
            cv::dct(block, dctBlock);
            embedDCT(dctBlock, bits, now); // now 會在函數內部被修改
            cv::idct(dctBlock, block);
        }
    }

    // 轉換回 CV_8U 並裁剪
    Mat final_stego;
    stego.convertTo(final_stego, CV_8U); // 轉換，會自動截斷 [0, 255]
    Mat cropped = final_stego(cv::Rect(0, 0, originalCols, originalRows)); // 使用原始尺寸裁剪

    // 儲存嵌入後的圖片
    cv::imshow("Original Image", img);
    cv::imshow("Stego Image", cropped); // 僅視覺化用的
    cv::imwrite(stegoImagePath, cropped);


    // 讀取剛剛儲存的隱寫圖片
    // Mat stego_img_loaded = cv::imread(stegoImagePath, cv::IMREAD_GRAYSCALE);
    Mat stego_img_loaded = stego.clone(); // 使用之前的圖片作為示範

    cout << "已讀取隱寫圖片: " << stegoImagePath << " (" << stego_img_loaded.rows << "x" << stego_img_loaded.cols << ")" << "\n";

    // 同樣需要補邊和轉換 (因為提取基於區塊)
    // 使用與嵌入時相同的 n, m 維度
    Mat Padded_stego;
    cv::copyMakeBorder(stego_img_loaded, Padded_stego, 0, n - originalRows, 0, m - originalCols, cv::BORDER_REPLICATE);
    Mat floatImg_stego;
    Padded_stego.convertTo(floatImg_stego, CV_32F);

    // 執行提取
    vector<bool> extracted_bits_all; // 儲存所有提取的位元
    bool foundNull = false;

    for (int i = 0; i < n && !foundNull; i += 8) { // 使用維度 n
        for (int j = 0; j < m && !foundNull; j += 8) { // 使用維度 m
            Mat block = floatImg_stego(cv::Rect(j, i, 8, 8));
            Mat dctBlock;
            cv::dct(block, dctBlock);
            vector<bool> block_bits = extractDCT(dctBlock);

            // 逐位元添加並檢查 null
            for (bool bit : block_bits) {
                extracted_bits_all.push_back(bit);
                if (extracted_bits_all.size() % 8 == 0 && extracted_bits_all.size() > 0) {
                    char lastChar = 0;
                    for(int k = 0; k < 8; ++k) {
                         lastChar = (lastChar << 1) | extracted_bits_all[extracted_bits_all.size() - 8 + k];
                    }
                    if (lastChar == '\0') {
                        foundNull = true;
                        extracted_bits_all.resize(extracted_bits_all.size() - 8); // 移除 null 位元
                        break; // 停止添加這個區塊的剩餘位元
                    }
                }
            }
        } 
    }
    // 轉換位元回字串
    string extracted_msg = bitstos(extracted_bits_all);
    cout << "Extracted Message: " << extracted_msg << "" << "\n";
    cv::waitKey(0);
    cv::destroyAllWindows();   
    return 0;
}