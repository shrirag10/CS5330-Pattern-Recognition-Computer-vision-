#pragma once
#include "Featurizer.h"

class BaselineFeaturizer : public Featurizer {
public:
    std::vector<float> featurize(const cv::Mat& image) override {
        int cx = image.cols / 2;
        int cy = image.rows / 2;

        std::vector<float> features;
        for (int r = cy - 3; r <= cy + 3; r++) {
            for (int c = cx - 3; c <= cx + 3; c++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);
                features.push_back(pixel[0]); // B
                features.push_back(pixel[1]); // G
                features.push_back(pixel[2]); // R
            }
        }
        return features; // 147 values
    }
};

class HistogramFeaturizer : public Featurizer {
public:
    HistogramFeaturizer(int bins = 16) : bins(bins) {}

    std::vector<float> featurize(const cv::Mat& image) override {
        float binWidth = 256.0f / bins;
        int total = image.rows * image.cols;

        // 3D histogram: hist[b][g][r] = count of pixels in that bin combination
        std::vector<float> hist(bins * bins * bins, 0.0f);

        for (int r = 0; r < image.rows; r++) {
            for (int c = 0; c < image.cols; c++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);
                int bBin = std::min((int)(pixel[0] / binWidth), bins - 1);
                int gBin = std::min((int)(pixel[1] / binWidth), bins - 1);
                int rBin = std::min((int)(pixel[2] / binWidth), bins - 1);
                hist[bBin * bins * bins + gBin * bins + rBin]++;
            }
        }

        // normalize by total pixels
        for (float& v : hist) v /= total;

        return hist; // bins^3 values (default: 512)
    }

private:
    int bins;
};
