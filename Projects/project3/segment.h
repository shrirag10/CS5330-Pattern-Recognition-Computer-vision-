#ifndef SEGMENT_H
#define SEGMENT_H

#include <opencv2/opencv.hpp>

/**
 * @brief Segment a binary image into regions using a custom 8-connected flood fill labeling from scratch.
 *        Regions smaller than minArea (in pixels) are filtered out and reset to background (0).
 * 
 * @param src Input binary image (CV_8UC1).
 * @param labelImg Output label image (CV_32SC1) containing 0 for background and 1, 2, ... for regions.
 * @param minArea Minimum pixel area for a region to be preserved.
 * @return Number of preserved regions (label IDs from 1 up to returning value).
 */
int segmentRegions(const cv::Mat &src, cv::Mat &labelImg, int minArea = 500);

/**
 * @brief Renders a color-coded image representing the labeled regions.
 *        Background (0) is rendered black. Each unique region ID gets a distinct color.
 * 
 * @param labelImg Input label image (CV_32SC1).
 * @param dst Output colorized image (CV_8UC3).
 * @param numRegions Total number of regions.
 */
void colorizeRegions(const cv::Mat &labelImg, cv::Mat &dst, int numRegions);

#endif // SEGMENT_H
