#include "cleanup.h"
#include <vector>
#include <algorithm>

// Custom Erosion from scratch
void customErosion(const cv::Mat &src, cv::Mat &dst, int elementSize) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.type() != CV_8UC1) {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
    }
    
    int radius = elementSize / 2;
    
    for (int r = 0; r < src.rows; ++r) {
        uchar *dstPtr = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c) {
            bool allSet = true;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = r + ky;
                if (ny < 0 || ny >= src.rows) {
                    allSet = false;
                    break;
                }
                const uchar *srcRow = src.ptr<uchar>(ny);
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = c + kx;
                    if (nx < 0 || nx >= src.cols) {
                        allSet = false;
                        break;
                    }
                    if (srcRow[nx] == 0) {
                        allSet = false;
                        break;
                    }
                }
                if (!allSet) break;
            }
            dstPtr[c] = allSet ? 255 : 0;
        }
    }
}

// Custom Dilation from scratch
void customDilation(const cv::Mat &src, cv::Mat &dst, int elementSize) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.type() != CV_8UC1) {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
    }
    
    int radius = elementSize / 2;
    
    for (int r = 0; r < src.rows; ++r) {
        uchar *dstPtr = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c) {
            bool anySet = false;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = r + ky;
                if (ny < 0 || ny >= src.rows) continue;
                const uchar *srcRow = src.ptr<uchar>(ny);
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = c + kx;
                    if (nx < 0 || nx >= src.cols) continue;
                    if (srcRow[nx] == 255) {
                        anySet = true;
                        break;
                    }
                }
                if (anySet) break;
            }
            dstPtr[c] = anySet ? 255 : 0;
        }
    }
}

// Custom Median Blur from scratch
void customMedianBlur(const cv::Mat &src, cv::Mat &dst, int kSize) {
    if (dst.rows != src.rows || dst.cols != src.cols || dst.type() != CV_8UC1) {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
    }
    
    int radius = kSize / 2;
    std::vector<uchar> window(kSize * kSize);
    
    for (int r = 0; r < src.rows; ++r) {
        uchar *dstPtr = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c) {
            int count = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = std::clamp(r + ky, 0, src.rows - 1);
                const uchar *srcRow = src.ptr<uchar>(ny);
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = std::clamp(c + kx, 0, src.cols - 1);
                    window[count++] = srcRow[nx];
                }
            }
            std::sort(window.begin(), window.begin() + count);
            dstPtr[c] = window[count / 2];
        }
    }
}

void cleanupBinary(const cv::Mat &src, cv::Mat &dst, int medianKernelSize, int morphElementSize) {
    if (src.empty()) {
        dst = src.clone();
        return;
    }

    cv::Mat temp = src.clone();

    // 1. Custom Median Filtering from scratch
    if (medianKernelSize > 0) {
        int kSize = medianKernelSize;
        if (kSize % 2 == 0) kSize += 1;
        cv::Mat blurred;
        customMedianBlur(temp, blurred, kSize);
        temp = blurred;
    }

    // 2. Custom Morphological Operations from scratch
    if (morphElementSize > 0) {
        cv::Mat closed, opened;
        
        // Custom Closing: Dilation followed by Erosion to fill internal holes
        cv::Mat dilated;
        customDilation(temp, dilated, morphElementSize);
        customErosion(dilated, closed, morphElementSize);

        // Custom Opening: Erosion followed by Dilation to remove background noise
        customErosion(closed, temp, morphElementSize);
        customDilation(temp, opened, morphElementSize);
        
        opened.copyTo(dst);
    } else {
        temp.copyTo(dst);
    }
}
