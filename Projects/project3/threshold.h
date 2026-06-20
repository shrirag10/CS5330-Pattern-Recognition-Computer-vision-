#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>

/**
 * @brief Custom binary thresholding from scratch.
 *        Takes a BGR or single-channel image, converts/processes it to grayscale,
 *        and thresholds it into a binary (8U) image.
 * 
 * @param src Source image (BGR or Grayscale).
 * @param dst Destination binary image (CV_8UC1).
 * @param thresholdValue The threshold limit (0-255).
 * @param invert If true, maps pixels below threshold to 255 (dark objects on light background).
 *               If false, maps pixels above threshold to 255.
 */
void thresholdBinary(const cv::Mat &src, cv::Mat &dst, int thresholdValue, bool invert = true);

/**
 * @brief Computes a dynamic threshold using the ISODATA (K-means with K=2) algorithm from scratch.
 *        To run in real-time, it samples a subset of the image's pixels.
 * 
 * @param src Source image (BGR or Grayscale).
 * @param sampleStride Stride for pixel sampling (e.g., 4 means sample every 4th pixel in x and y).
 * @return int The calculated threshold value (0-255).
 */
int computeDynamicThresholdISODATA(const cv::Mat &src, int sampleStride = 4);

#endif // THRESHOLD_H
