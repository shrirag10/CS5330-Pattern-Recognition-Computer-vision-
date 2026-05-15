/*
 * CS5330 Computer Vision - Project 1
 * Authors: [Your Name], Shyam S (shyams612)
 * Date: May 2026
 *
 * Purpose: Task 1 — Reads an image file from the command line, resizes it
 *          to 1080x720, displays it in a window, and waits for 'q' to quit.
 *
 * Usage: ./imgDisplay <image_path>
 */

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./imgDisplay <image_path>" << std::endl;
        return -1;
    }

    // Read the image from disk
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cout << "Could not read image: " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "Original size: " << img.cols << "x" << img.rows << std::endl;

    // Resize for consistent display
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(1080, 720));
    std::cout << "Resized to: " << resized.cols << "x" << resized.rows << std::endl;

    cv::namedWindow("Image", 1);
    cv::imshow("Image", resized);

    // Wait in a loop; 'q' exits
    while (true) {
        char key = (char)cv::waitKey(10);
        if (key == 'q') break;
    }

    cv::destroyAllWindows();
    return 0;
}