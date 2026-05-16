/*
 * CS5330 Computer Vision - Project 1
 * Authors: [Your Name], Shyam S (shyams612)
 * Date: May 2026
 *
 * Purpose: Unified live video display program supporting Tasks 2-9.
 *          Opens the default webcam and applies selected filters based
 *          on the last keypress. Pressing the same key again returns to
 *          normal (unfiltered) color video.
 *
 * Keymap:
 *   q - quit
 *   s - save current (processed) frame to a timestamped PNG
 *   g - Task 3: toggle OpenCV cvtColor greyscale
 *   h - Task 4: toggle custom blue-emphasis greyscale
 *   p - Task 5: toggle sepia tone
 *   b - Task 6: toggle 5x5 blur (blur5x5_2)
 *   1 - Task 6: run blur5x5_1 (naive) with timing printed to console
 *   x - Task 7: toggle Sobel X (absolute value displayed)
 *   y - Task 7: toggle Sobel Y (absolute value displayed)
 *   m - Task 8: toggle gradient magnitude
 *   l - Task 9: toggle blur+quantize (10 levels)
 *   f - Task 10: toggle Haar cascade face detection with bounding boxes
 */

#include <iostream>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include <vector>
#include "faceDetect.h"

int main(int argc, char *argv[]) {

    // Open the default video capture device
    cv::VideoCapture *capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return -1;
    }

    // Print captured frame dimensions
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    printf("Press 'q' to quit, 's' to save frame.\n");
    printf("Keys: g=opencv-grey, h=custom-grey, p=sepia, b=blur,\n");
    printf("      1=timed-blur1, x=sobelX, y=sobelY, m=magnitude, l=blur+quantize,\n");
    printf("      f=face-detect\n");

    cv::namedWindow("Video", 1);
    std::filesystem::create_directories("saved");

    cv::Mat frame;
    char mode = ' ';  // currently active filter; ' ' = no filter
    cv::Mat grey;
    std::vector<cv::Rect> faces;
    for (;;) {
        *capdev >> frame;
        if (frame.empty()) {
            printf("frame is empty\n");
            break;
        }

        cv::Mat display;  // what gets shown in the window

        // Apply the selected filter based on current mode
        if (mode == 'g') {
            // Task 3: OpenCV standard greyscale (cvtColor)
            // OpenCV weights: 0.114B + 0.587G + 0.299R (ITU-R BT.601 luma)
            cv::Mat grey;
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
            cv::cvtColor(grey, display, cv::COLOR_GRAY2BGR);

        } else if (mode == 'h') {
            // Task 4: Custom blue-emphasis greyscale (see filter.cpp)
            greyscale(frame, display);

        } else if (mode == 'p') {
            // Task 5: Sepia tone
            sepia(frame, display);

        } else if (mode == 'b') {
            // Task 6: Fast separable 5x5 blur
            auto t0 = std::chrono::high_resolution_clock::now();
            blur5x5_2(frame, display);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("blur5x5_2: %.2f ms\n", ms);

        } else if (mode == '1') {
            // Task 6: Naive 5x5 blur (slow, for timing comparison)
            auto t0 = std::chrono::high_resolution_clock::now();
            blur5x5_1(frame, display);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("blur5x5_1: %.2f ms\n", ms);

        } else if (mode == 'x') {
            // Task 7: Sobel X — show absolute value as a displayable image
            cv::Mat sx;
            sobelX3x3(frame, sx);
            cv::convertScaleAbs(sx, display);  // converts 16S → 8U

        } else if (mode == 'y') {
            // Task 7: Sobel Y — show absolute value as a displayable image
            cv::Mat sy;
            sobelY3x3(frame, sy);
            cv::convertScaleAbs(sy, display);

        } else if (mode == 'm') {
            // Task 8: Gradient magnitude from Sobel X and Y
            cv::Mat sx, sy;
            sobelX3x3(frame, sx);
            sobelY3x3(frame, sy);
            magnitude(sx, sy, display);

        } else if (mode == 'l') {
            // Task 9: Blur + quantize (10 levels for a cartoon/poster effect)
            blurQuantize(frame, display, 10);

        } 
        else if(mode == 'f') {
            display = frame.clone();
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
            detectFaces(grey, faces);
            drawBoxes(display, faces);
        }
        else {
            // Default: show original color frame
            display = frame;
        }

        cv::imshow("Video", display);

        // Check for keypress (10ms wait)
        char key = (char)cv::waitKey(10);

        if (key == 'q') {
            break;  // quit

        } else if (key == 's') {
            // Save the current processed frame with a timestamp
            std::time_t t = std::time(nullptr);
            char timestamp[32];
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S",
                          std::localtime(&t));
            std::string filename = std::string("saved/frame_") + timestamp + ".png";
            cv::imwrite(filename, display);
            printf("Saved: %s\n", filename.c_str());

        } else if (key != -1) {
            // Any other key: toggle mode (same key twice returns to normal)
            mode = (mode == key) ? ' ' : key;
            printf("Mode: %c\n", mode == ' ' ? '-' : mode);
        }
    }

    delete capdev;
    cv::destroyAllWindows();
    return 0;
}