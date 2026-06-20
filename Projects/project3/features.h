#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

struct RegionFeatures {
    int labelId;
    double area;               // m00
    cv::Point2d centroid;      // (x_bar, y_bar)
    double mu20;               // central moments
    double mu02;
    double mu11;
    double orientation;        // primary axis orientation in radians
    double elongation;         // max / min moments of inertia (ratio of eigenvalues)
    std::vector<double> featureVec; // feature vector: [elongation, h1, h2]
};

/**
 * @brief Computes centroid, central moments, orientation, elongation, and feature vectors for each region.
 * 
 * @param labelImg Input label image (CV_32SC1). Background is 0, labeled regions are 1, 2, ...
 * @param numRegions Number of regions in labelImg.
 * @return std::vector<RegionFeatures> Vector of features for valid regions (label > 0).
 */
std::vector<RegionFeatures> computeRegionFeatures(const cv::Mat &labelImg, int numRegions);

/**
 * @brief Overlay centroid crosshair, primary axis line, and text statistics for each region on the image.
 * 
 * @param dst Input/Output color image to draw overlays on.
 * @param features Vector of computed region features.
 */
void drawRegionFeatures(cv::Mat &dst, const std::vector<RegionFeatures> &features);

#endif // FEATURES_H
