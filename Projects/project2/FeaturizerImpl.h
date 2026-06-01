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
