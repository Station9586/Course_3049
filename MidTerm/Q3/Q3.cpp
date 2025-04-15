#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

// --- 參數設定 ---
const int CODEBOOK_SIZE = 128;       // 目標碼書大小
const int BLOCK_SIZE = 4;           // 影像區塊邊長 (4x4)
const int VECTOR_DIM = BLOCK_SIZE * BLOCK_SIZE; // 向量維度 (16)
const double KMEANS_EPSILON = 1e-5;   // K-means 迭代收斂閾值 (相對失真變化)
const int MAX_KMEANS_ITERATIONS = 100;// K-means 最大迭代次數
const double SPLIT_PERTURBATION = 1.0; // 分裂時的擾動值

// 從影像檔案載入訓練向量
bool loadTrainingVectors(const std::vector<std::string>& imagePaths, int blockSize, std::vector<cv::Mat>& trainingVectors) {
    trainingVectors.clear();
    int vectorDim = blockSize * blockSize;

    for (const std::string& path : imagePaths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "無法載入訓練影像 '" << path << "'" << "\n";
            continue; // 跳過這個檔案
        }

        // 確保影像是灰階
        if (img.channels() != 1) {
             cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        // 確保資料類型為 CV_8U (雖然我們會轉成 CV_32F 計算，但載入時確認一下)
        if (img.type() != CV_8U) {
            img.convertTo(img, CV_8U);
        }

        // 確保影像尺寸可以被 blockSize 整除
        int rows = (img.rows / blockSize) * blockSize;
        int cols = (img.cols / blockSize) * blockSize;
        if (rows != img.rows || cols != img.cols) {
            std::cout << "提示：影像 '" << path << "' 尺寸 (" << img.cols << "x" << img.rows
                      << ") 無法被區塊大小 " << blockSize << " 整除，將裁切至 ("
                      << cols << "x" << rows << ")" << "\n";
            img = img(cv::Rect(0, 0, cols, rows));
        }


        // 提取區塊並轉換為向量
        for (int r = 0; r < img.rows; r += blockSize) {
            for (int c = 0; c < img.cols; c += blockSize) {
                cv::Mat block = img(cv::Rect(c, r, blockSize, blockSize));
                cv::Mat block_float;
                block.convertTo(block_float, CV_32F); // 轉換為浮點數進行計算
                // 將區塊攤平成 1xN 的向量 (N = vectorDim)
                cv::Mat vector = block_float.reshape(1, 1);
                trainingVectors.push_back(vector.clone()); // 加入 clone 以免指向 ROI
            }
        }
        // std::cout << "已處理影像: " << path << "\n";
    }
    return !trainingVectors.empty(); // 如果至少有一個向量被提取，則返回 true
}

// 計算一組向量的質心 (平均向量)
cv::Mat calculateCentroid(const std::vector<cv::Mat>& vectors) {
    if (vectors.empty()) {
        return cv::Mat(); // 返回空 Mat
    }
    // 假設所有向量維度相同，取第一個向量的維度
    int vectorDim = vectors[0].cols;
    int numVectors = vectors.size();

    // 初始化總和向量為零
    cv::Mat sumVector = cv::Mat::zeros(1, vectorDim, CV_32F);

    // 累加所有向量
    for (const cv::Mat& vec : vectors) {
        // 確保維度一致 (可以增加錯誤檢查)
        if (vec.cols == vectorDim && vec.rows == 1 && vec.type() == CV_32F) {
             sumVector += vec;
        } else {
            std::cerr << "向量維度或類型不符，已跳過。" << "\n";
        }
    }

    // 計算平均值
    cv::Mat centroid = sumVector / static_cast<double>(numVectors);
    return centroid;
}

// LBG 演算法主體
std::vector<cv::Mat> trainLBG(const std::vector<cv::Mat>& trainingVectors, int targetCodebookSize, int vectorDim, double epsilon, int maxIterations, double perturbation) {
    if (trainingVectors.empty()) {
        return {}; // 返回空碼書
    }

    // --- 1. 初始化碼書 (大小為 1) ---
    std::vector<cv::Mat> codebook;
    codebook.push_back(calculateCentroid(trainingVectors)); // 初始碼向量是所有訓練資料的平均值
    std::cout << "  初始碼書大小: 1" << "\n";

    double lastAvgDistortion = std::numeric_limits<double>::max();

    // --- 2. 迭代增長碼書 ---
    while (codebook.size() < targetCodebookSize) {
        // --- 2a. 分裂碼書 ---
        std::vector<cv::Mat> newCodebook;
        cv::Mat perturbationVector = cv::Mat::ones(1, vectorDim, CV_32F) * perturbation; // 建立擾動向量
        perturbationVector.at<float>(0,0) += 0.1; // 稍微打破對稱性

        std::cout << "  分裂碼書從 " << codebook.size() << " 到 ";
        for (const cv::Mat& codeword : codebook) {
            if (newCodebook.size() < targetCodebookSize) {
                newCodebook.push_back(codeword + perturbationVector);
            }
            if (newCodebook.size() < targetCodebookSize) {
                 newCodebook.push_back(codeword - perturbationVector);
            }
        }
        codebook = newCodebook;
        std::cout << codebook.size() << "..." << "\n";


        // --- 2b. K-means 迭代優化當前碼書 ---
        std::cout << "    執行 K-means 優化 (目標大小 " << codebook.size() << "):" << "\n";
        for (int iter = 0; iter < maxIterations; ++iter) {
            // -- 分配步驟 --
            std::vector<std::vector<int>> clusters(codebook.size()); // 儲存每個聚類包含的訓練向量索引
            double currentTotalDistortion = 0.0;

            for (int i = 0; i < trainingVectors.size(); ++i) {
                double minDistSq = std::numeric_limits<double>::max();
                int nearestCodewordIndex = -1;

                // 找到最近的碼向量
                for (int k = 0; k < codebook.size(); ++k) {
                    // 使用平方歐氏距離以提高效率
                    double distSq = cv::norm(trainingVectors[i], codebook[k], cv::NORM_L2SQR);
                    if (distSq < minDistSq) {
                        minDistSq = distSq;
                        nearestCodewordIndex = k;
                    }
                }

                if (nearestCodewordIndex != -1) {
                    clusters[nearestCodewordIndex].push_back(i);
                    currentTotalDistortion += minDistSq;
                }
            }

            // -- 更新步驟 --
            int emptyClusters = 0;
            for (int k = 0; k < codebook.size(); ++k) {
                if (!clusters[k].empty()) {
                    // 計算新質心
                    std::vector<cv::Mat> clusterVectors;
                    for (int index : clusters[k]) {
                        clusterVectors.push_back(trainingVectors[index]);
                    }
                    codebook[k] = calculateCentroid(clusterVectors);
                } else {
                    emptyClusters++;
                }
            }
             if (emptyClusters > 0) {
                 std::cerr << "    警告：K-means 迭代 " << iter << " 發現 " << emptyClusters << " 個空聚類！" << "\n";
             }


            // -- 檢查收斂 --
            double avgDistortion = currentTotalDistortion / trainingVectors.size();
            double distortionChange = std::abs(lastAvgDistortion - avgDistortion);

            std::cout << "      迭代 " << iter << ": 平均失真 = " << avgDistortion
                      << ", 變化 = " << distortionChange << "\n";

            // 使用相對變化量判斷收斂
            if (lastAvgDistortion != 0 && (distortionChange / lastAvgDistortion) < epsilon) {
                 std::cout << "    K-means 收斂於迭代 " << iter << "\n";
                break; // 收斂，跳出 K-means 迭代
            }
            lastAvgDistortion = avgDistortion;

            if (iter == maxIterations - 1) {
                std::cout << "    達到最大 K-means 迭代次數。" << "\n";
            }
        } // K-means 迭代結束
         lastAvgDistortion = std::numeric_limits<double>::max(); // 重置失真，準備下一次分裂後的 K-means
    } // 碼書增長迴圈結束

    return codebook;
}

// 將碼書儲存到檔案 (使用 OpenCV FileStorage)
bool saveCodebook(const std::string& filename, const std::vector<cv::Mat>& codebook) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "無法開啟檔案 '" << filename << "' 進行寫入。" << "\n";
        return false;
    }

    fs << "codebookSize" << (int)codebook.size();
    if (!codebook.empty()) {
         fs << "vectorDim" << codebook[0].cols;
         fs << "blockSize" << (int)std::sqrt(codebook[0].cols); // 假設是方形區塊
    } else {
         fs << "vectorDim" << 0;
         fs << "blockSize" << 0;
    }


    fs << "codewords" << "[";
    for (const auto& codeword : codebook) {
        fs << codeword;
    }
    fs << "]";

    fs.release(); // 關閉檔案
    return true;
}


// 將碼書視覺化 (將每個碼向量顯示為影像區塊)
cv::Mat visualizeCodebook(const std::vector<cv::Mat>& codebook, int blockSize) {
    if (codebook.empty()) {
        return cv::Mat();
    }

    int codebookSize = codebook.size();
    int vectorDim = codebook[0].cols;

    // 計算網格布局，盡量接近方形
    int gridCols = static_cast<int>(std::ceil(std::sqrt(codebookSize)));
    int gridRows = static_cast<int>(std::ceil(static_cast<double>(codebookSize) / gridCols));

    // 計算視覺化影像的總尺寸
    int totalWidth = gridCols * blockSize;
    int totalHeight = gridRows * blockSize;

    // 創建一個黑色背景的影像來放置所有碼向量區塊
    // 初始設為 CV_32F，方便處理碼向量的浮點值
    cv::Mat visualization = cv::Mat::zeros(totalHeight, totalWidth, CV_32F);

    double minVal, maxVal; // 用於歸一化

    // 找到所有碼向量中的最小和最大值，用於歸一化顯示
    cv::Mat allCodewords;
    cv::vconcat(codebook, allCodewords); // 將所有碼向量垂直串接成一個大矩陣
    cv::minMaxLoc(allCodewords, &minVal, &maxVal);

    // 將每個碼向量 reshape 成區塊，歸一化並複製到視覺化影像中
    for (int i = 0; i < codebookSize; ++i) {
        // 計算當前碼向量在網格中的位置
        int gridX = (i % gridCols) * blockSize;
        int gridY = (i / gridCols) * blockSize;

        // Reshape 碼向量回 blockSize x blockSize 的區塊
        cv::Mat block = codebook[i].clone().reshape(1, blockSize);

        // 歸一化到 0-255 範圍以便顯示
        cv::Mat normalizedBlock;
        if (maxVal > minVal) {
            // 使用 minMaxLoc 找到的全局最小最大值進行歸一化
             block.convertTo(normalizedBlock, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        } else {
            // 如果所有值都一樣，則設為中間灰度值
            block.convertTo(normalizedBlock, CV_8U); // 先轉成 8U
            normalizedBlock.setTo(cv::Scalar(128));
        }


        // 複製歸一化後的區塊到視覺化影像的對應位置
        cv::Rect roi(gridX, gridY, blockSize, blockSize);
        normalizedBlock.copyTo(visualization(roi));
    }

     // 最終將整個視覺化影像轉為 CV_8U
     cv::Mat finalVisualization;
     if (maxVal > minVal) {
          visualization.convertTo(finalVisualization, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
     } else {
          visualization.convertTo(finalVisualization, CV_8U);
          finalVisualization.setTo(cv::Scalar(128));
     }

    return finalVisualization;
}

// --- 主函數 ---
int main (void) {
    std::cout << "LBG VQ 碼書訓練程式" << "\n";
    std::cout << "目標碼書大小: " << CODEBOOK_SIZE << "\n";
    std::cout << "影像區塊大小: " << BLOCK_SIZE << "x" << BLOCK_SIZE << " (維度: " << VECTOR_DIM << ")" << "\n";

    const std::string train_path = "train_img/image";

    std::vector<std::string> imagePaths;
    for (int i = 1; i <= 10; ++i) imagePaths.emplace_back(train_path + std::to_string(i) + ".png");


    std::cout << "使用的訓練影像數量: " << imagePaths.size() << "\n";

    // --- 載入訓練向量 ---
    std::vector<cv::Mat> trainingVectors;
    if (!loadTrainingVectors(imagePaths, BLOCK_SIZE, trainingVectors)) {
        std::cerr << "載入訓練向量失敗。" << "\n";
        return -1;
    }
    if (trainingVectors.empty()) {
        std::cerr << "沒有從影像中提取到任何訓練向量。" << "\n";
        return -1;
    }
    std::cout << "從訓練影像中提取了 " << trainingVectors.size() << " 個訓練向量。" << "\n";

    // --- 執行 LBG 演算法訓練碼書 ---
    std::cout << "開始訓練 LBG 碼書..." << "\n";
    std::vector<cv::Mat> codebook = trainLBG(trainingVectors, CODEBOOK_SIZE, VECTOR_DIM, KMEANS_EPSILON, MAX_KMEANS_ITERATIONS, SPLIT_PERTURBATION);
    std::cout << "LBG 訓練完成，最終碼書大小: " << codebook.size() << "\n";

    // --- 儲存碼書 ---
    std::string codebookFilename = "lbg_codebook_" + std::to_string(CODEBOOK_SIZE) + ".yml";
    if (saveCodebook(codebookFilename, codebook)) {
        // std::cout << "碼書已儲存到檔案: " << codebookFilename << "\n";
    } else {
        std::cerr << "儲存碼書失敗。" << "\n";
    }

    // --- 視覺化碼書 ---
    cv::Mat codebookVisualization = visualizeCodebook(codebook, BLOCK_SIZE);
    if (!codebookVisualization.empty()) {
        std::string visualizationFilename = "lbg_codebook_visualization_" + std::to_string(CODEBOOK_SIZE) + ".png";
        cv::imwrite(visualizationFilename, codebookVisualization);
        cv::imshow("LBG Codebook Visualization", codebookVisualization);
        cv::waitKey(0);
    } else {
        std::cerr << "產生碼書視覺化影像失敗。" << "\n";
    }

    cv::destroyAllWindows();
    return 0;
}