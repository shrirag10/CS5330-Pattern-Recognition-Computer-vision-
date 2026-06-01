#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class Featurizer {
public:
    virtual std::vector<float> featurize(const cv::Mat& image) = 0;

    std::vector<std::vector<float>> featurizeAll(const std::vector<cv::Mat>& images) {
        std::vector<std::vector<float>> results;
        for (const auto& img : images)
            results.push_back(featurize(img));
        return results;
    }

    virtual ~Featurizer() = default;
};
