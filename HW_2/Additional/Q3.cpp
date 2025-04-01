#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

// --- Configuration ---
const int SUBPIXEL_BLOCK_SIZE = 2; // Using 2x2 subpixels per secret pixel
const uchar BLACK_PIXEL = 0;
const uchar WHITE_PIXEL = 255;

const uchar W = WHITE_PIXEL, B = BLACK_PIXEL;
// --- Define the 6 valid 2x2 patterns (2 black, 2 white) ---
// B = BLACK_PIXEL, W = WHITE_PIXEL
const vector<vector<vector<uchar>>> patterns = {
    {{B, W}, {W, B}},
    {{W, B}, {B, W}},
    {{B, B}, {W, W}},
    {{W, W}, {B, B}},
    {{B, W}, {B, W}}, // Vertical stripes
    {{W, B}, {W, B}}  // Vertical stripes complement
    // Can add more patterns if needed, as long as they have 2 black, 2 white
};

// --- Helper function to get the complement of a pattern ---
vector<vector<uchar>> getComplementPattern(const vector<vector<uchar>>& pattern) {
    vector<vector<uchar>> complement = pattern;
    for (int i = 0; i < SUBPIXEL_BLOCK_SIZE; ++i) {
        for (int j = 0; j < SUBPIXEL_BLOCK_SIZE; ++j) {
            complement[i][j] = (pattern[i][j] == BLACK_PIXEL) ? WHITE_PIXEL : BLACK_PIXEL;
        }
    }
    return complement;
}

// --- Helper function to set a 2x2 subpixel block in a share image ---
void setSubPixelBlock(Mat& share, int startX, int startY, const vector<vector<uchar>>& pattern) {
    for (int i = 0; i < SUBPIXEL_BLOCK_SIZE; ++i) {
        for (int j = 0; j < SUBPIXEL_BLOCK_SIZE; ++j) {
            // Ensure we don't write outside bounds (though sizes should match)
            if (startY + i < share.rows && startX + j < share.cols) {
                 share.at<uchar>(startY + i, startX + j) = pattern[i][j];
            }
        }
    }
}

int main() {
    // --- 1. Load Secret Image ---
    string secretImagePath = "../img/image.png";
    Mat secretImage = imread(secretImagePath, IMREAD_GRAYSCALE);

    // --- 2. Ensure Image is Binary (Black and White) ---
    // Use thresholding. Adjust threshold value if needed (128 is common midpoint)
    Mat binarySecretImage;
    threshold(secretImage, binarySecretImage, 128, 255, THRESH_BINARY);

    // --- 3. Initialize Shares ---
    int secretRows = binarySecretImage.rows;
    int secretCols = binarySecretImage.cols;
    int shareRows = secretRows * SUBPIXEL_BLOCK_SIZE;
    int shareCols = secretCols * SUBPIXEL_BLOCK_SIZE;

    Mat share1 = Mat::zeros(shareRows, shareCols, CV_8U); // Initialize shares
    Mat share2 = Mat::zeros(shareRows, shareCols, CV_8U);

    // --- 4. Setup Random Number Generation ---
    // Use <random> for better randomness than rand()
    std::mt19937 rng(static_cast<unsigned int>(time(0))); // Mersenne Twister engine seeded with time
    std::uniform_int_distribution<int> dist(0, patterns.size() - 1); // Distribution for pattern index

    // --- 5. Generate Shares ---
    cout << "Generating shares..." << "\n";
    for (int y = 0; y < secretRows; ++y) for (int x = 0; x < secretCols; ++x) {
        uchar secretPixel = binarySecretImage.at<uchar>(y, x);
        int patternIndex = dist(rng); // Choose a random pattern index

        // Get the chosen pattern
        const vector<vector<uchar>>& chosenPattern = patterns[patternIndex];

        // Calculate starting coordinates in the shares
        int startShareY = y * SUBPIXEL_BLOCK_SIZE;
        int startShareX = x * SUBPIXEL_BLOCK_SIZE;

        if (secretPixel == WHITE_PIXEL) {
            // Rule for WHITE: Both shares get the SAME random pattern
            setSubPixelBlock(share1, startShareX, startShareY, chosenPattern);
            setSubPixelBlock(share2, startShareX, startShareY, chosenPattern);
        } else { // secretPixel == BLACK_PIXEL
            // Rule for BLACK: Share1 gets the pattern, Share2 gets its COMPLEMENT
            vector<vector<uchar>> complementPattern = getComplementPattern(chosenPattern);
            setSubPixelBlock(share1, startShareX, startShareY, chosenPattern);
            setSubPixelBlock(share2, startShareX, startShareY, complementPattern);
        }
    }


    // --- 6. Simulate Overlay ---
    // Physical overlay simulation: If either share pixel is black, overlay is black.
    // Equivalent to pixel-wise MIN operation when Black=0, White=255.
    Mat overlayedResult = Mat::zeros(shareRows, shareCols, CV_8U);
    min(share1, share2, overlayedResult); // Pixel-wise minimum


    // --- 7. Display and Save Results ---
    imshow("Binarized Secret Image", binarySecretImage);
    imshow("Share 1 (Noise)", share1);
    imshow("Share 2 (Noise)", share2);
    imshow("Overlayed Result (Secret Revealed)", overlayedResult);

    // Save the generated images
    imwrite("Q3_secret_binary.png", binarySecretImage);
    imwrite("Q3_share1.png", share1);
    imwrite("Q3_share2.png", share2);
    imwrite("Q3_overlayed_result.png", overlayedResult);

    waitKey(0);
    destroyAllWindows();

    return 0;
}