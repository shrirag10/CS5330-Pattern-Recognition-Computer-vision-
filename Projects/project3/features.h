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
    
    // OBB bounds (Task 4)
    double minE1;
    double maxE1;
    double minE2;
    double maxE2;

    // Classification result (Task 9)
    std::string className;
    double classDist;
};

// Embedding Database Instance for Task 9
struct EmbeddingInstance {
    std::string label;
    std::vector<float> embedding;
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

/**
 * @brief Appends a training label and its features to a CSV database file.
 * 
 * @param dbPath Path to the CSV database file.
 * @param label The category/class name of the object.
 * @param features The feature vector to write.
 * @return true if successfully saved, false otherwise.
 */
bool saveTrainingInstance(const std::string &dbPath, const std::string &label, const std::vector<double> &features);

// Embedding Database Helpers (Task 9)
bool saveEmbeddingInstance(const std::string &dbPath, const std::string &label, const cv::Mat &embedding);
std::vector<EmbeddingInstance> loadEmbeddingDatabase(const std::string &dbPath);
double computeCosineDistance(const cv::Mat &embA, const std::vector<float> &embB);

// External Utilities from utilities.cpp (Task 9)
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,
                         float minE1, float maxE1, float minE2, float maxE2, int debug);

#endif // FEATURES_H
