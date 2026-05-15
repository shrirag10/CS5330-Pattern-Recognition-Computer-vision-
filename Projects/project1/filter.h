/*
 * CS5330 Computer Vision - Project 1
 * Authors: [Your Name], Shyam S (shyams612)
 * Date: May 2026
 *
 * Purpose: Declares all image processing and filter functions for Project 1.
 *          Implementations are in filter.cpp.
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

/**
 * Custom greyscale using blue-emphasis weighting (0.15R + 0.35G + 0.50B).
 * Produces a cooler, visually distinct result compared to OpenCV's cvtColor.
 * All three output channels are set to the same computed grey value.
 *
 * @param src  Input color image (CV_8UC3, BGR)
 * @param dst  Output greyscale-as-color image (CV_8UC3, all channels equal)
 * @return 0 on success, -1 on error
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

/**
 * Sepia tone filter making the image look antique.
 * Each output channel is a weighted sum of all three input channels (matrix below).
 * Output values are clamped to [0, 255]. Original channel values are used for all
 * three new-channel computations (no in-place channel reuse).
 *
 *   new_B = 0.272*R + 0.534*G + 0.131*B
 *   new_G = 0.349*R + 0.686*G + 0.168*B
 *   new_R = 0.393*R + 0.769*G + 0.189*B
 *
 * @param src  Input color image (CV_8UC3, BGR)
 * @param dst  Output sepia image (CV_8UC3, BGR)
 * @return 0 on success, -1 on error
 */
int sepia(cv::Mat &src, cv::Mat &dst);

/**
 * Naive 5x5 Gaussian blur using the .at<> pixel accessor.
 * Kernel: [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1] (sum=100).
 * Outer 2 rows/cols are copied from src unchanged.
 *
 * @param src  Input color image (CV_8UC3, BGR) — not modified
 * @param dst  Output blurred image (CV_8UC3, BGR)
 * @return 0 on success, -1 on error
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/**
 * Fast 5x5 Gaussian blur using separable 1x5 passes and row pointers.
 * Applies filter [1 2 4 2 1] (sum=10) horizontally, then vertically.
 * Avoids the slow .at<> accessor. Outer 2 rows/cols copied from src.
 *
 * @param src  Input color image (CV_8UC3, BGR) — not modified
 * @param dst  Output blurred image (CV_8UC3, BGR)
 * @return 0 on success, -1 on error
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

/**
 * 3x3 Sobel X filter implemented as two separable 1x3 passes.
 * Detects vertical edges. Positive values = right-going gradient.
 * Horizontal: [-1 0 1], Vertical: [1 2 1].
 * Output is 16-bit signed to accommodate values in [-1020, 1020].
 *
 * @param src  Input color image (CV_8UC3, BGR)
 * @param dst  Output gradient image (CV_16SC3)
 * @return 0 on success, -1 on error
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * 3x3 Sobel Y filter implemented as two separable 1x3 passes.
 * Detects horizontal edges. Positive values = upward-going gradient.
 * Horizontal: [1 2 1], Vertical: [1 0 -1].
 * Output is 16-bit signed to accommodate values in [-1020, 1020].
 *
 * @param src  Input color image (CV_8UC3, BGR)
 * @param dst  Output gradient image (CV_16SC3)
 * @return 0 on success, -1 on error
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

/**
 * Computes per-channel gradient magnitude from Sobel X and Y images.
 * Formula: dst = sqrt(sx*sx + sy*sy), clamped to [0, 255].
 *
 * @param sx   Sobel X image (CV_16SC3)
 * @param sy   Sobel Y image (CV_16SC3)
 * @param dst  Output magnitude image (CV_8UC3, BGR)
 * @return 0 on success, -1 on error
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/**
 * Blurs a color image then quantizes each channel into `levels` discrete values.
 * Quantization: bucket = 255/levels; new_val = (val / bucket) * bucket.
 * Uses blur5x5_2 internally.
 *
 * @param src     Input color image (CV_8UC3, BGR)
 * @param dst     Output blurred+quantized image (CV_8UC3, BGR)
 * @param levels  Number of quantization levels per channel (e.g. 10)
 * @return 0 on success, -1 on error
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

#endif // FILTER_H
