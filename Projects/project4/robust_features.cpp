#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    // Parse arguments
    bool headless = false;
    std::string input_source = "";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else {
            input_source = arg;
        }
    }

    // Determine input source
    cv::VideoCapture cap;
    bool is_webcam = true;
    std::vector<std::string> image_paths;
    int current_img_idx = 0;

    if (!input_source.empty()) {
        if (std::filesystem::is_directory(input_source)) {
            is_webcam = false;
            for (const auto &entry : std::filesystem::directory_iterator(input_source)) {
                std::string ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".JPG") {
                    image_paths.push_back(entry.path().string());
                }
            }
            std::sort(image_paths.begin(), image_paths.end());
            if (image_paths.empty()) {
                std::cerr << "Error: No images found in directory " << input_source << std::endl;
                return -1;
            }
            std::cout << "Loaded " << image_paths.size() << " images for feature detection." << std::endl;
        } else {
            is_webcam = false;
            if (!cap.open(input_source)) {
                std::cerr << "Error: Could not open video file " << input_source << std::endl;
                return -1;
            }
            std::cout << "Opened video file: " << input_source << std::endl;
        }
    } else {
        if (headless) {
            std::cerr << "Error: Headless mode requires an input directory or video file." << std::endl;
            return -1;
        }
        if (!cap.open(0)) {
            std::cerr << "Error: Could not open webcam." << std::endl;
            return -1;
        }
        std::cout << "Opened default webcam." << std::endl;
    }

    // Parameters for feature detection
    int detector_type = 0; // 0 = Harris, 1 = ORB
    int harris_threshold = 120; // range 0-255
    int orb_max_features = 500;

    if (!headless) {
        cv::namedWindow("Robust Feature Detection", cv::WINDOW_AUTOSIZE);
        // Create trackbars for interactive tuning
        cv::createTrackbar("Detector (0:Harris, 1:ORB)", "Robust Feature Detection", &detector_type, 1);
        cv::createTrackbar("Harris Thresh", "Robust Feature Detection", &harris_threshold, 255);
    } else {
        std::filesystem::create_directories("data/ar_output");
    }

    std::cout << "\n--- Feature Detection Controls ---\n"
              << "Press 't' to toggle between Harris corners and ORB keypoints.\n"
              << "Press 'q' to quit.\n"
              << "----------------------------------\n" << std::endl;

    cv::Mat frame;
    while (true) {
        if (is_webcam || cap.isOpened()) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }
        } else {
            if (current_img_idx >= static_cast<int>(image_paths.size())) {
                break;
            }
            frame = cv::imread(image_paths[current_img_idx]);
            if (frame.empty()) {
                current_img_idx++;
                continue;
            }
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::Mat display_frame = frame.clone();

        if (detector_type == 0) {
            // 1. Harris Corner Detection
            cv::Mat dst = cv::Mat::zeros(frame.size(), CV_32FC1);
            int blockSize = 2;
            int apertureSize = 3;
            double k = 0.04;

            cv::cornerHarris(gray, dst, blockSize, apertureSize, k);

            // Normalize
            cv::Mat dst_norm, dst_norm_scaled;
            cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
            cv::convertScaleAbs(dst_norm, dst_norm_scaled);

            // Draw corners
            int count = 0;
            for (int j = 0; j < dst_norm.rows; j++) {
                for (int i = 0; i < dst_norm.cols; i++) {
                    if (static_cast<int>(dst_norm.at<float>(j, i)) > harris_threshold) {
                        cv::circle(display_frame, cv::Point(i, j), 5, cv::Scalar(0, 0, 255), 2);
                        count++;
                    }
                }
            }
            std::string label = "Harris Corners (Thresh: " + std::to_string(harris_threshold) + "): " + std::to_string(count);
            cv::putText(display_frame, label, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        } else {
            // 2. ORB Feature Detection
            cv::Ptr<cv::ORB> orb = cv::ORB::create(orb_max_features);
            std::vector<cv::KeyPoint> keypoints;
            orb->detect(gray, keypoints);

            // Draw rich keypoints
            cv::drawKeypoints(display_frame, keypoints, display_frame, cv::Scalar(255, 0, 0), 
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            std::string label = "ORB Keypoints (Count: " + std::to_string(keypoints.size()) + ")";
            cv::putText(display_frame, label, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        if (!headless) {
            cv::imshow("Robust Feature Detection", display_frame);
            char key = static_cast<char>(cv::waitKey(is_webcam ? 30 : 0));
            if (key == 'q') {
                break;
            } else if (key == 't') {
                detector_type = 1 - detector_type;
                cv::setTrackbarPos("Detector (0:Harris, 1:ORB)", "Robust Feature Detection", detector_type);
                std::cout << "Switched to " << (detector_type == 0 ? "Harris Corner" : "ORB Keypoint") << " detector." << std::endl;
            }
        } else {
            // In headless mode, save both versions for testing
            std::string Harris_fn = "data/ar_output/features_harris_frame_" + std::to_string(current_img_idx + 1) + ".png";
            std::string ORB_fn = "data/ar_output/features_orb_frame_" + std::to_string(current_img_idx + 1) + ".png";

            // Process Harris and save
            cv::Mat dst = cv::Mat::zeros(frame.size(), CV_32FC1);
            cv::cornerHarris(gray, dst, 2, 3, 0.04);
            cv::Mat dst_norm;
            cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
            cv::Mat display_harris = frame.clone();
            for (int j = 0; j < dst_norm.rows; j++) {
                for (int i = 0; i < dst_norm.cols; i++) {
                    if (static_cast<int>(dst_norm.at<float>(j, i)) > harris_threshold) {
                        cv::circle(display_harris, cv::Point(i, j), 5, cv::Scalar(0, 0, 255), 2);
                    }
                }
            }
            cv::imwrite(Harris_fn, display_harris);
            std::cout << "Saved: " << Harris_fn << std::endl;

            // Process ORB and save
            cv::Ptr<cv::ORB> orb = cv::ORB::create(orb_max_features);
            std::vector<cv::KeyPoint> keypoints;
            orb->detect(gray, keypoints);
            cv::Mat display_orb = frame.clone();
            cv::drawKeypoints(display_orb, keypoints, display_orb, cv::Scalar(255, 0, 0), 
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imwrite(ORB_fn, display_orb);
            std::cout << "Saved: " << ORB_fn << std::endl;
        }

        if (!is_webcam && !cap.isOpened()) {
            current_img_idx++;
        }
    }

    if (!headless) {
        cv::destroyAllWindows();
    }
    return 0;
}
