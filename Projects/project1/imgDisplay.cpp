#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./imgDisplay <image_path>" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cout << "Could not read image: " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "Original size: " << img.cols << "x" << img.rows << std::endl;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(1080, 720));

    std::cout << "Resized to: " << resized.cols << "x" << resized.rows << std::endl;
    cv::imshow("Resized", resized);
    cv::waitKey(0);

    return 0;
}