#pragma once
#include <string>
#include <opencv2/opencv.hpp>

struct NamedImage {
    std::string name;
    cv::Mat mat;
};
