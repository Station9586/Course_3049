#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

// Finds the largest dimension divisible by 2^levels below the original size
int adjust_dim(int dim, int levels) {
    int factor = 1 << levels; // 2^levels
    return (dim / factor) * factor;
}

// --- Function to perform 1-level 2D Haar DWT ---
// Input: src (CV_32F grayscale image)
// Outputs: ll, lh, hl, hh (CV_32F subbands)
void haar_dwt_2d(const Mat& src, Mat& ll, Mat& lh, Mat& hl, Mat& hh) {
    if (src.empty() || src.type() != CV_32F) {
        cerr << "Error: Input matrix for haar_dwt_2d must be CV_32F and non-empty." << "\n";
        return;
    }
    if (src.rows % 2 != 0 || src.cols % 2 != 0) {
         cerr << "Error: Input matrix dimensions must be even for Haar DWT." << "\n";
         return;
    }

    int rows = src.rows;
    int cols = src.cols;
    int half_rows = rows / 2;
    int half_cols = cols / 2;

    // Temporary matrices for row transform
    Mat temp_l(rows, half_cols, CV_32F);
    Mat temp_h(rows, half_cols, CV_32F);

    // 1. Row transform (Horizontal)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < half_cols; ++j) {
            float p1 = src.at<float>(i, 2 * j);
            float p2 = src.at<float>(i, 2 * j + 1);
            temp_l.at<float>(i, j) = (p1 + p2) / 2.0f; // Average (Low pass)
            temp_h.at<float>(i, j) = (p1 - p2) / 2.0f; // Difference (High pass)
            // Using sqrt(2) scaling factor is theoretically correct,
            // but /2 is common in image processing for range.
            // temp_l.at<float>(i, j) = (p1 + p2) / sqrt(2.0f);
            // temp_h.at<float>(i, j) = (p1 - p2) / sqrt(2.0f);
        }
    }

    // Allocate output matrices
    ll = Mat(half_rows, half_cols, CV_32F);
    lh = Mat(half_rows, half_cols, CV_32F);
    hl = Mat(half_rows, half_cols, CV_32F);
    hh = Mat(half_rows, half_cols, CV_32F);

    // 2. Column transform (Vertical)
    for (int j = 0; j < half_cols; ++j) {
        // On Low-pass columns from row transform
        for (int i = 0; i < half_rows; ++i) {
            float p1 = temp_l.at<float>(2 * i, j);
            float p2 = temp_l.at<float>(2 * i + 1, j);
            ll.at<float>(i, j) = (p1 + p2) / 2.0f; // LL (Avg of Avg)
            lh.at<float>(i, j) = (p1 - p2) / 2.0f; // LH (Diff of Avg)
            // ll.at<float>(i, j) = (p1 + p2) / sqrt(2.0f);
            // lh.at<float>(i, j) = (p1 - p2) / sqrt(2.0f);
        }
        // On High-pass columns from row transform
        for (int i = 0; i < half_rows; ++i) {
            float p1 = temp_h.at<float>(2 * i, j);
            float p2 = temp_h.at<float>(2 * i + 1, j);
            hl.at<float>(i, j) = (p1 + p2) / 2.0f; // HL (Avg of Diff)
            hh.at<float>(i, j) = (p1 - p2) / 2.0f; // HH (Diff of Diff)
            // hl.at<float>(i, j) = (p1 + p2) / sqrt(2.0f);
            // hh.at<float>(i, j) = (p1 - p2) / sqrt(2.0f);
        }
    }
}

// --- Helper function to normalize and display a subband ---
void display_subband(const Mat& subband, const string& window_name, bool normalize_for_display = true) {
    if (subband.empty()) {
        cerr << "Warning: Trying to display empty subband: " << window_name << "\n";
        return;
    }
    Mat display_mat;
    if (normalize_for_display) {
        // Normalize to 0-255 for visualization, especially for LH, HL, HH
         // Method 1: Use normalize function
        normalize(subband, display_mat, 0, 255, NORM_MINMAX, CV_8U);

        // Method 2: Add offset (useful if data is centered around 0)
        // Mat shifted;
        // subband.convertTo(shifted, CV_32F, 1.0, 128.0); // Shift range
        // shifted.convertTo(display_mat, CV_8U); // Convert to 8U
    } else {
        // For LL band, just convert type (assuming it resembles original image)
        subband.convertTo(display_mat, CV_8U);
    }
    imshow(window_name, display_mat);
}


int main (void) {
    // --- 1. Load Image ---
    string image_path = "../img/image.png";
    Mat img = imread(image_path, IMREAD_GRAYSCALE);

    // --- 2. Preprocessing ---
    int levels = 3;
    int new_rows = adjust_dim(img.rows, levels);
    int new_cols = adjust_dim(img.cols, levels);

    if (new_rows == 0 || new_cols == 0) {
        cerr << "Error: Image dimensions too small for " << levels << " levels of DWT." << "\n";
        return -1;
    }

    Mat resized_img;
    resize(img, resized_img, Size(new_cols, new_rows));

    Mat float_img;
    resized_img.convertTo(float_img, CV_32F); // Convert to float for calculations

    // --- 3. Multi-level Haar DWT ---
    vector<Mat> ll_levels, lh_levels, hl_levels, hh_levels;
    Mat current_ll = float_img.clone(); // Start with the full (resized) image

    for (int level = 1; level <= levels; ++level) {
        Mat ll, lh, hl, hh;
        haar_dwt_2d(current_ll, ll, lh, hl, hh);

        // Store the subbands for this level
        ll_levels.push_back(ll);
        lh_levels.push_back(lh);
        hl_levels.push_back(hl);
        hh_levels.push_back(hh);

        // Display subbands for the current level
        string level_str = "Level " + to_string(level) + " ";
        display_subband(ll, level_str + "LL", false); // LL doesn't usually need normalization like others
        display_subband(lh, level_str + "LH");
        display_subband(hl, level_str + "HL");
        display_subband(hh, level_str + "HH");

        // Update the input for the next level
        current_ll = ll; // Next level processes the current LL band
    }

    // --- 4. Create Composite Visualization Image ---
    Mat dwt_display = Mat::zeros(float_img.size(), CV_8U); // Create black canvas

    // Place the final LL subband (LL3) in the top-left corner
    Mat final_ll_display;
    ll_levels.back().convertTo(final_ll_display, CV_8U); // Convert LL3 to 8U
    final_ll_display.copyTo(dwt_display(Rect(0, 0, final_ll_display.cols, final_ll_display.rows)));

    // Place the LH, HL, HH subbands from highest level to lowest
    int current_row_offset = 0;
    int current_col_offset = 0;
    for (int level = levels - 1; level >= 0; --level) {
        Mat lh_norm, hl_norm, hh_norm;
        int sub_cols = lh_levels[level].cols;
        int sub_rows = lh_levels[level].rows;

        normalize(lh_levels[level], lh_norm, 0, 255, NORM_MINMAX, CV_8U);
        normalize(hl_levels[level], hl_norm, 0, 255, NORM_MINMAX, CV_8U);
        normalize(hh_levels[level], hh_norm, 0, 255, NORM_MINMAX, CV_8U);

        // Calculate top-left corner for this level's subbands
        // For level 'l', offset is based on size of LL_(l+1) if l < levels-1
        // For the highest level (l = levels-1), offset is (0, size_ll) for HL, (size_ll, 0) for LH, (size_ll, size_ll) for HH
        if (level == levels - 1) { // Highest level subbands placement relative to LL_N
           current_col_offset = sub_cols; // HL starts right of LL_N
           current_row_offset = 0;
           hl_norm.copyTo(dwt_display(Rect(current_col_offset, current_row_offset, sub_cols, sub_rows)));

           current_col_offset = 0;        // LH starts below LL_N
           current_row_offset = sub_rows;
           lh_norm.copyTo(dwt_display(Rect(current_col_offset, current_row_offset, sub_cols, sub_rows)));

           current_col_offset = sub_cols; // HH starts diagonal to LL_N
           current_row_offset = sub_rows;
           hh_norm.copyTo(dwt_display(Rect(current_col_offset, current_row_offset, sub_cols, sub_rows)));

        } else { // Lower levels placement relative to the whole quadrant
            current_col_offset = sub_cols; // HL is in top-right quadrant of its level block
            current_row_offset = 0;
             hl_norm.copyTo(dwt_display(Rect(current_col_offset, current_row_offset, sub_cols, sub_rows)));

             current_col_offset = 0;       // LH is in bottom-left quadrant
             current_row_offset = sub_rows;
             lh_norm.copyTo(dwt_display(Rect(current_col_offset, current_row_offset, sub_cols, sub_rows)));

             current_col_offset = sub_cols; // HH is in bottom-right quadrant
             current_row_offset = sub_rows;
             hh_norm.copyTo(dwt_display(Rect(current_col_offset, current_row_offset, sub_cols, sub_rows)));
        }
    }


    // --- 5. Display Final Results ---
    imshow("Original Resized Image", resized_img);
    imshow("Composite Haar DWT (3 Levels)", dwt_display);
    imwrite("Q1_Composite_DWT.png", dwt_display);
    waitKey(0);
    destroyAllWindows();

    return 0;
}