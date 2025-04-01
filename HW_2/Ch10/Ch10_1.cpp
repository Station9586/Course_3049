#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using cv::Mat;

// 將字串轉換成位元 vector
vector<bool> stobits (const string &s) {
    vector<bool> bits; // 存放轉換後的位元
    bits.reserve(s.length() << 3); // 預先保留空間 (字串長度 * 8)

    // 遍歷字串中的每個字元
    for (char c: s) {
        // 遍歷字元的每個位元 (從最高位元到最低位元)
        for (int i = 7; ~i; --i) {
            bits.push_back((c >> i) & 1); // 取出第 i 個位元並加入 vector
        }
    }
    return bits; // 回傳位元 vector
}

// 將位元 vector 轉換回字串
string bitstos (const vector<bool> &bits) {
    string s = ""; // 存放轉換後的字串
    char now = 0; // 當前正在組合的字元
    int cnt = 0; // 目前已組合的位元數

    s.reserve(bits.size() >> 3); // 預先保留空間 (位元數量 / 8)

    // 遍歷位元 vector
    for (bool b: bits) {
        now = (now << 1) | b; // 將當前字元左移一位，並將新的位元加到最低位
        ++cnt; // 位元計數加一

        // 如果已組合 8 個位元
        if (cnt == 8) {
            if (now == '\0') break; // 如果組合出空字元 (字串結束符)，則停止轉換
            s += now; // 將組合好的字元附加到字串
            now = 0; // 重置當前字元
            cnt = 0; // 重置位元計數
        }
    }

    return s; // 回傳轉換後的字串
}

// 將訊息嵌入封面影像中 (LSB 隱寫)
// cover: 原始封面影像
// stego: 輸出，嵌入訊息後的影像
// msg: 要嵌入的訊息字串
bool embed (const Mat &cover, Mat &stego, const string &msg) {
    string s = msg + '\0'; // 在訊息尾端加上空字元作為結束標記
    vector<bool> bits = stobits(s); // 將含結束標記的訊息轉換成位元
    int n = cover.rows, m = cover.cols, N = bits.size(); // 獲取影像尺寸和位元數量

    // 檢查訊息位元數是否超過影像可容納的量 (每個像素的 BGR 三個通道各藏 1 bit)
    if (bits.size() > (long long)n * m * 3) return false; // 容量不足，嵌入失敗

    stego = cover.clone(); // 複製一份封面影像，用於修改

    int id = 0; // 目前處理到位元 vector 的索引
    // 遍歷影像的每個像素
    for (int i = 0; i < n and id < N; ++i) for (int j = 0; j < m and id < N; ++j) {
        cv::Vec3b &pixel = stego.at<cv::Vec3b>(i, j); // 取得像素的 BGR 值 (注意是 Vec3b&，可以修改原值)
        // 遍歷像素的 B, G, R 三個通道
        for (int k = 0; k < 3 and id < N; ++k) {
            // 將通道值的最低位元 (LSB) 替換成訊息位元
            // pixel[k] & 0xFE: 將最低位元清零 (AND 11111110)
            // | bits[id++]: 將訊息位元設定到最低位 (OR 0000000x)
            pixel[k] = (pixel[k] & 0xFE) | bits[id++];
        }
    }

    return true; // 嵌入成功
}

// 從隱寫影像中提取訊息
// stego: 含有隱藏訊息的影像
string extract (const Mat &stego) {
    int n = stego.rows, m = stego.cols; // 獲取影像尺寸

    vector<bool> bits; // 存放提取出的位元
    bool ok = false; // 標記是否已找到訊息結尾 (空字元)

    // 遍歷影像的每個像素
    for (int i = 0; i < n and !ok; ++i) for (int j = 0; j < m and !ok; ++j) {
        cv::Vec3b pixel = stego.at<cv::Vec3b>(i, j); // 取得像素的 BGR 值 (這次是 Vec3b，非參考)
        // 遍歷像素的 B, G, R 三個通道
        for (int k = 0; k < 3 and !ok; ++k) {
            bits.push_back(pixel[k] & 1); // 提取通道值的最低位元 (LSB) 並加入 vector
            // 每提取 8 個位元，檢查一次是否組成了空字元
            if (bits.size() % 8 == 0 && bits.size() > 0) { // 確保至少有8個位元
                char last = 0; // 用於組合最後 8 個位元
                // 從 bits vector 的尾部取出 8 個位元來組合成字元
                for (int l = 0; l < 8; ++l) {
                    last = (last << 1) | bits[bits.size() - 8 + l];
                }

                // 如果組合成空字元，表示訊息結束
                if (last == '\0') {
                    ok = true; // 設定找到結尾的標記
                    bits.resize(bits.size() - 8); // 從 bits vector 中移除最後代表空字元的 8 個位元
                }
            }
        }
    }

    return bitstos(bits); // 將提取出的位元轉換回字串
}

int main (void) {
    Mat image = cv::imread("../img/image.png"); // 讀取封面影像
    if (image.empty()) { // 檢查影像是否成功讀取
        cout << "Error loading image." << "\n";
        return 1;
    }
    string s = "Hello, World!"; // 要隱藏的訊息
    Mat stego; // 用於存放隱寫後的影像

    // 嘗試嵌入訊息
    if (embed(image, stego, s)) {
        cout << "Message embedded successfully.\n"; // 嵌入成功
    } else {
        cout << "Failed to embed message. Image capacity might be insufficient.\n"; // 嵌入失敗
        return 1;
    }

    // 將帶有隱藏訊息的影像存檔
    cv::imwrite("ch10_1_stego_image.png", stego);

    // 從隱寫影像中提取訊息
    string extracted = extract(stego);
    cout << "Extracted message: " << extracted << "\n"; // 輸出提取的訊息

    // 驗證提取的訊息是否與原始訊息相同
    if (extracted == s) {
        cout << "Message extracted successfully.\n"; // 提取成功且內容正確
    } else {
        cout << "Failed to extract message correctly.\n"; // 提取失敗或內容錯誤
        return 1;
    }

    // 顯示原始影像和隱寫後的影像 (需要圖形介面支援)
    cv::imshow("Original", image);
    cv::imshow("Stego", stego);
    cv::waitKey(0); // 等待使用者按任意鍵
    cv::destroyAllWindows(); // 關閉所有 OpenCV 視窗
    return 0;
}