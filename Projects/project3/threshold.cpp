#include "threshold.h"
#include <vector>
#include <algorithm>
#include <cmath>

// Custom 3x3 box blur from scratch to smooth out noise and make regions uniform
void customBoxBlurGrayscale(const cv::Mat &src, cv::Mat &dst) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.type() != CV_8UC1) {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
    }
    
    // Copy borders
    src.copyTo(dst);
    
    // Perform 3x3 box blur on interior pixels
    for (int r = 1; r < src.rows - 1; ++r) {
        const uchar *prevRow = src.ptr<uchar>(r - 1);
        const uchar *currRow = src.ptr<uchar>(r);
        const uchar *nextRow = src.ptr<uchar>(r + 1);
        uchar *dstRow = dst.ptr<uchar>(r);
        
        for (int c = 1; c < src.cols - 1; ++c) {
            int sum = prevRow[c - 1] + prevRow[c] + prevRow[c + 1] +
                      currRow[c - 1] + currRow[c] + currRow[c + 1] +
                      nextRow[c - 1] + nextRow[c] + nextRow[c + 1];
            dstRow[c] = sum / 9;
        }
    }
}

// Unified pre-processing: Grayscale conversion + Saturation discount + Custom Box Blur
void preprocessGrayscale(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat gray = cv::Mat::zeros(src.size(), CV_8UC1);
    int channels = src.channels();
    
    for (int r = 0; r < src.rows; ++r) {
        const uchar *srcPtr = src.ptr<uchar>(r);
        uchar *grayPtr = gray.ptr<uchar>(r);
        
        for (int c = 0; c < src.cols; ++c) {
            if (channels == 3) {
                uchar b = srcPtr[c * 3 + 0];
                uchar g = srcPtr[c * 3 + 1];
                uchar r_val = srcPtr[c * 3 + 2];
                
                // 0.299R + 0.587G + 0.114B luma weighting
                float intensity = 0.299f * r_val + 0.587f * g + 0.114f * b;
                
                // Saturation: max - min
                uchar max_c = std::max({r_val, g, b});
                uchar min_c = std::min({r_val, g, b});
                float saturation = max_c - min_c;
                
                // Subtract saturation to darken colored regions
                float customIntensity = intensity - 0.5f * saturation;
                grayPtr[c] = std::clamp(static_cast<int>(customIntensity), 0, 255);
            } else {
                grayPtr[c] = srcPtr[c];
            }
        }
    }
    
    // Apply blur to make regions more uniform
    customBoxBlurGrayscale(gray, dst);
}

void thresholdBinary(const cv::Mat &src, cv::Mat &dst, int thresholdValue, bool invert) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.type() != CV_8UC1) {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
    }
    
    // Pre-process (Grayscale + Saturation discount + Blur)
    cv::Mat preprocessed;
    preprocessGrayscale(src, preprocessed);
    
    // Apply threshold
    for (int r = 0; r < preprocessed.rows; ++r) {
        const uchar *prepPtr = preprocessed.ptr<uchar>(r);
        uchar *dstPtr = dst.ptr<uchar>(r);
        for (int c = 0; c < preprocessed.cols; ++c) {
            if (invert) {
                dstPtr[c] = (prepPtr[c] < thresholdValue) ? 255 : 0;
            } else {
                dstPtr[c] = (prepPtr[c] >= thresholdValue) ? 255 : 0;
            }
        }
    }
}

int computeDynamicThresholdISODATA(const cv::Mat &src, int sampleStride) {
    cv::Mat preprocessed;
    preprocessGrayscale(src, preprocessed);
    
    std::vector<int> sampledPixels;
    sampledPixels.reserve((preprocessed.rows / sampleStride) * (preprocessed.cols / sampleStride));
    
    // Sample pixels from the preprocessed (blurred) image
    for (int r = 0; r < preprocessed.rows; r += sampleStride) {
        const uchar *prepPtr = preprocessed.ptr<uchar>(r);
        for (int c = 0; c < preprocessed.cols; c += sampleStride) {
            sampledPixels.push_back(prepPtr[c]);
        }
    }
    
    if (sampledPixels.empty()) {
        return 128; // Fallback
    }
    
    // Calculate initial T (mean)
    double sum = 0;
    for (int val : sampledPixels) {
        sum += val;
    }
    double T = sum / sampledPixels.size();
    
    // Iterate until convergence
    const int maxIterations = 100;
    for (int iter = 0; iter < maxIterations; ++iter) {
        double bgSum = 0;
        int bgCount = 0;
        double fgSum = 0;
        int fgCount = 0;
        
        for (int val : sampledPixels) {
            if (val < T) {
                bgSum += val;
                bgCount++;
            } else {
                fgSum += val;
                fgCount++;
            }
        }
        
        if (bgCount == 0 || fgCount == 0) {
            break;
        }
        
        double bgMean = bgSum / bgCount;
        double fgMean = fgSum / fgCount;
        double nextT = (bgMean + fgMean) / 2.0;
        
        if (std::abs(nextT - T) < 0.5) {
            T = nextT;
            break;
        }
        T = nextT;
    }
    
    return std::clamp(static_cast<int>(std::round(T)), 0, 255);
}
