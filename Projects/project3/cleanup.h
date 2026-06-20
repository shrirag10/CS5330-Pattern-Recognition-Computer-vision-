#ifndef CLEANUP_H
#define CLEANUP_H

#include <opencv2/opencv.hpp>

/**
 * @brief Cleans up a binary thresholded image using median filtering and morphological operations.
 *        1. Applies a median blur to filter isolated high-frequency pixel noise (salt-and-pepper).
 *        2. Performs morphological closing (dilation then erosion) to fill internal holes in objects.
 *        3. Performs morphological opening (erosion then dilation) to remove boundary noise/spurs.
 * 
 * @param src Input binary image (CV_8UC1).
 * @param dst Output cleaned binary image (CV_8UC1).
 * @param medianKernelSize Kernel size for median filter (must be positive and odd, e.g., 3 or 5).
 *                         If <= 0, median filtering is skipped.
 * @param morphElementSize Size of the rectangular structuring element (e.g., 3 or 5).
 *                         If <= 0, morphological operations are skipped.
 */
void cleanupBinary(const cv::Mat &src, cv::Mat &dst, int medianKernelSize = 5, int morphElementSize = 3);

#endif // CLEANUP_H
