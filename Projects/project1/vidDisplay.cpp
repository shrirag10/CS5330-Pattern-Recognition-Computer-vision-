/*
 * CS5330 Computer Vision - Project 1
 * Authors: Shriman Raghav Srinivasan, Shyam Srinivasan
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
 *   d - Task 11: toggle depth map visualization (INFERNO colormap)
 *   c - Task 11: toggle depth-based background greyscale (creative filter)
 *   n - Task 12: toggle color negative
 *   e - Task 13: toggle emboss effect
 *   o - Task 14: toggle depth-based fog
 */

#include <iostream>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include <vector>
#include "faceDetect.h"
#include "da2-code/DA2Network.hpp"

int main(int argc, char *argv[]) {

    // Load DA2 depth network once before the main loop
    DA2Network da_net("model_fp16.onnx");

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
    printf("      f=face-detect, d=depth-map, c=depth-defocus, n=negative, e=emboss, o=fog\n");

    // scale factor: normalize shorter side to 256px for the depth network
    float da2_scale = 256.0f / std::min(refS.width, refS.height);

    cv::namedWindow("Video", 1);
    std::filesystem::create_directories("../saved");

    cv::Mat frame;
    char mode = ' ';  // currently active filter; ' ' = no filter
    cv::Mat grey;
    cv::Mat depth;    // grayscale depth output from DA2
    std::vector<cv::Rect> faces;

    // mouse callback: click anywhere on the video window to print depth value at that pixel
    cv::setMouseCallback("Video", [](int event, int x, int y, int, void *userdata) {
        if (event != cv::EVENT_LBUTTONDOWN) return;
        cv::Mat *dep = static_cast<cv::Mat *>(userdata);
        if (dep->empty()) {
            printf("Depth map not available (press 'd' or 'c' first)\n");
            return;
        }
        if (x < 0 || x >= dep->cols || y < 0 || y >= dep->rows) return;
        int val = dep->at<unsigned char>(y, x);
        printf("Depth at (%d, %d): %d  [0=closest, 255=farthest]\n", x, y, val);
    }, &depth);
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

        } else if (mode == 'n') {
            // Task 12: color negative
            negative(frame, display);

        } else if (mode == 'e') {
            // Task 13: emboss effect
            emboss(frame, display);

        } else if (mode == 'o') {
            // Task 14: depth-based exponential fog
            da_net.set_input(frame, da2_scale);
            da_net.run_network(depth, frame.size());
            depthFog(frame, depth, display);

        } else if (mode == 'd') {
            // Task 11: depth map visualization
            da_net.set_input(frame, da2_scale);
            da_net.run_network(depth, frame.size());
            cv::applyColorMap(depth, display, cv::COLORMAP_INFERNO);

        } else if (mode == 'c') {
            // Task 11: depth-based background greyscale
            // foreground (depth <= threshold) stays in color; background goes greyscale
            da_net.set_input(frame, da2_scale);
            da_net.run_network(depth, frame.size());

            cv::Mat grey_bgr;
            cv::Mat grey_single;
            cv::cvtColor(frame, grey_single, cv::COLOR_BGR2GRAY);
            cv::cvtColor(grey_single, grey_bgr, cv::COLOR_GRAY2BGR);

            display = frame.clone();
            const unsigned char threshold = 150; // higher = closer in DA2 output
            for (int i = 0; i < frame.rows; i++) {
                const cv::Vec3b *src_row    = frame.ptr<cv::Vec3b>(i);
                const cv::Vec3b *grey_row   = grey_bgr.ptr<cv::Vec3b>(i);
                const unsigned char *dep_row = depth.ptr<unsigned char>(i);
                cv::Vec3b *dst_row          = display.ptr<cv::Vec3b>(i);
                for (int j = 0; j < frame.cols; j++) {
                    dst_row[j] = (dep_row[j] >= threshold) ? src_row[j] : grey_row[j];
                }
            }
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
            std::string filename = std::string("../saved/frame_") + timestamp + ".png";
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