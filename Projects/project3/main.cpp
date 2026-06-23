#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "features.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    std::cout << "========================================================\n"
              << "          CS 5330 - Project 3: 2-D Object Recognition    \n"
              << "          Task 5: Training Mode & Database Setup         \n"
              << "========================================================\n"
              << "Controls:\n"
              << "  'q' : Quit the application\n"
              << "  't' : Toggle threshold view (on/off in main window)\n"
              << "  'd' : Toggle dynamic thresholding (ISODATA) vs manual\n"
              << "  'i' : Toggle threshold inversion (dark on light vs light on dark)\n"
              << "  '[' or ',' : Decrease threshold value by 5\n"
              << "  ']' or '.' : Increase threshold value by 5\n"
              << "  'c' : Toggle morphological cleanup (on/off)\n"
              << "  '-' : Decrease morphological structuring element size\n"
              << "  '=' : Increase morphological structuring element size\n"
              << "  's' : Save current frame & processed images to disk\n"
              << "  'r' : Record training sample to database.csv\n"
              << "  'n' : Next static image (only when camera is not used)\n"
              << "  'p' : Previous static image (only when camera is not used)\n"
              << "========================================================\n" << std::endl;

    // Set up list of development images for fallback/static mode
    std::vector<std::string> devImages = {
        "data/img1p3.png",
        "data/img2P3.png",
        "data/img3P3.png",
        "data/img4P3.png",
        "data/img5P3.png"
    };
    int currentImgIdx = 0;
    bool useCamera = true;

    // Check if an image path was provided on command line
    std::string argPath = "";
    bool testMode = false;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--test") {
            testMode = true;
            useCamera = false;
        } else {
            argPath = arg;
            useCamera = false;
        }
    }

    cv::VideoCapture cap;
    if (useCamera) {
        // Try opening default webcam
        cap.open(0);
        if (!cap.isOpened()) {
            std::cout << "[Warning] Webcam 0 could not be opened. Falling back to static development images." << std::endl;
            useCamera = false;
        }
    }

    // Create directory for saved outputs
    fs::create_directories("saved_thresholds");

    if (testMode) {
        std::cout << "[Test Mode] Running automated thresholding, cleanup, & segmentation..." << std::endl;
        for (size_t i = 0; i < devImages.size(); ++i) {
            std::string path = devImages[i];
            cv::Mat img = cv::imread(path);
            if (img.empty()) {
                std::cout << "[Error] Could not load image: " << path << std::endl;
                continue;
            }
            int activeThreshold = computeDynamicThresholdISODATA(img, 4);
            cv::Mat bin, clean;
            thresholdBinary(img, bin, activeThreshold, true); // invert = true
            cleanupBinary(bin, clean, 5, 3); // 5x5 median, 3x3 morph element

            cv::Mat labelImg, coloredRegions;
            int numRegions = segmentRegions(clean, labelImg, 500);
            colorizeRegions(labelImg, coloredRegions, numRegions);

            std::vector<RegionFeatures> features = computeRegionFeatures(labelImg, numRegions);
            cv::Mat featImg = img.clone();
            drawRegionFeatures(featImg, features);

            std::string outOrig = "saved_thresholds/orig_dev" + std::to_string(i + 1) + ".png";
            std::string outThresh = "saved_thresholds/thresh_dev" + std::to_string(i + 1) + ".png";
            std::string outClean = "saved_thresholds/clean_dev" + std::to_string(i + 1) + ".png";
            std::string outRegions = "saved_thresholds/regions_dev" + std::to_string(i + 1) + ".png";
            std::string outFeatures = "saved_thresholds/features_dev" + std::to_string(i + 1) + ".png";

            cv::imwrite(outOrig, img);
            cv::imwrite(outThresh, bin);
            cv::imwrite(outClean, clean);
            cv::imwrite(outRegions, coloredRegions);
            cv::imwrite(outFeatures, featImg);

            std::cout << "  - Processed " << path 
                      << " (Threshold: " << activeThreshold 
                      << ", Regions: " << numRegions 
                      << ") -> saved Region Map and Features" << std::endl;
        }
        std::cout << "[Test Mode] Done. Verification files saved to saved_thresholds/" << std::endl;
        return 0;
    }

    // Windows setup
    cv::namedWindow("Original Feed", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Thresholded View", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Regions Map", cv::WINDOW_AUTOSIZE);

    // load classification DB
    std::vector<TrainingEntry> trainingDB = loadDatabase("database.csv");
    if (trainingDB.empty()) {
        std::cout << "[Warning] No training data found. Run ./train first." << std::endl;
    } else {
        std::cout << "[Info] Loaded " << trainingDB.size() << " training entries." << std::endl;
    }

    int manualThreshold = 128;
    bool useDynamic = true;
    bool invertThreshold = true;
    bool showThresholdInMain = false;
    bool useCleanup = true;
    int medianKernelSize = 5;
    int morphElementSize = 3;

    cv::Mat frame, binary, cleaned, labelImg, coloredRegions;

    bool isTypingLabel = false;
    std::string typedLabel = "";

    while (true) {
        if (useCamera) {
            if (!isTypingLabel) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "[Error] Empty frame captured from camera. Exiting..." << std::endl;
                    break;
                }
            }
        } else {
            // Load from file list or command line argument
            std::string path = argPath.empty() ? devImages[currentImgIdx] : argPath;
            frame = cv::imread(path);
            if (frame.empty()) {
                std::cout << "[Error] Could not load image: " << path << std::endl;
                if (!argPath.empty()) {
                    break; // if user specified an invalid arg, exit
                }
                // Otherwise move to next available
                currentImgIdx = (currentImgIdx + 1) % devImages.size();
                continue;
            }
            // downscale large images to fit screen
            int maxDim = 800;
            if (frame.cols > maxDim || frame.rows > maxDim) {
                double scale = std::min((double)maxDim / frame.cols, (double)maxDim / frame.rows);
                cv::resize(frame, frame, cv::Size(), scale, scale);
            }
        }

        // Apply thresholding
        int activeThreshold = manualThreshold;
        if (useDynamic) {
            activeThreshold = computeDynamicThresholdISODATA(frame, 4);
        }

        thresholdBinary(frame, binary, activeThreshold, invertThreshold);

        // Apply cleanup
        if (useCleanup) {
            cleanupBinary(binary, cleaned, medianKernelSize, morphElementSize);
        } else {
            cleaned = binary.clone();
        }

        // Apply segmentation
        int numRegions = segmentRegions(cleaned, labelImg, 500);
        colorizeRegions(labelImg, coloredRegions, numRegions);

        // Compute features
        std::vector<RegionFeatures> features = computeRegionFeatures(labelImg, numRegions);

        // Display results
        cv::Mat displayFrame = frame.clone();
        
        // Draw feature overlays on the clean frame display
        drawRegionFeatures(displayFrame, features);

        // classify largest region and overlay label
        if (!features.empty() && !trainingDB.empty()) {
            RegionFeatures largest = features[0];
            for (const auto &rf : features)
                if (rf.area > largest.area) largest = rf;
            std::string label = classifyFeatures(largest.featureVec, trainingDB);
            cv::putText(displayFrame, label,
                        cv::Point((int)largest.centroid.x - 30, (int)largest.centroid.y - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }

        if (showThresholdInMain) {
            cv::cvtColor(cleaned, displayFrame, cv::COLOR_GRAY2BGR);
        }

        // Draw HUD overlay on display frame
        std::string modeStr = useDynamic ? "Dynamic (ISODATA)" : "Manual";
        std::string invStr = invertThreshold ? "Dark-on-Light" : "Light-on-Dark";
        std::string cleanStr = useCleanup ? "ON (SE=" + std::to_string(morphElementSize) + ")" : "OFF";
        
        std::string hudText = "Threshold: " + std::to_string(activeThreshold) + " | Mode: " + modeStr;
        std::string hudText2 = "Inversion: " + invStr + " | Cleanup: " + cleanStr;
        std::string hudText3 = "Regions: " + std::to_string(numRegions);
        if (!useCamera && argPath.empty()) {
            hudText2 += " | File: " + devImages[currentImgIdx];
        }

        cv::putText(displayFrame, hudText, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(displayFrame, hudText2, cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(displayFrame, hudText3, cv::Point(15, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // Draw Interactive Label GUI Overlay if typing
        if (isTypingLabel) {
            int h = displayFrame.rows;
            int w = displayFrame.cols;
            cv::Mat overlay = displayFrame.clone();
            // Dark transparent bar at bottom
            cv::rectangle(overlay, cv::Point(0, h - 70), cv::Point(w, h), cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.7, displayFrame, 0.3, 0.0, displayFrame);

            cv::putText(displayFrame, "TRAINING MODE: Enter object label name", cv::Point(15, h - 45), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            cv::putText(displayFrame, typedLabel + "_", cv::Point(15, h - 20), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            cv::putText(displayFrame, "(ENTER to Save, ESC to Cancel)", cv::Point(w - 280, h - 20), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        }

        cv::imshow("Original Feed", displayFrame);
        cv::imshow("Thresholded View", cleaned);
        cv::imshow("Regions Map", coloredRegions);

        // Wait 30ms or check key press
        char key = (char)cv::waitKey(30);

        if (isTypingLabel) {
            if (key == 27) { // ESC key
                isTypingLabel = false;
                std::cout << "[Training Mode] Canceled." << std::endl;
            } else if (key == 13 || key == 10) { // Enter key
                if (!typedLabel.empty()) {
                    if (features.empty()) {
                        std::cout << "[Training Mode] No regions detected in frame to train!" << std::endl;
                    } else {
                        // Find the largest region by area
                        RegionFeatures largest = features[0];
                        for (const auto &rf : features) {
                            if (rf.area > largest.area) {
                                largest = rf;
                            }
                        }
                        bool success = saveTrainingInstance("saved_thresholds/database.csv", typedLabel, largest.featureVec);
                        if (success) {
                            std::cout << "[Training Mode] Saved training instance of '" << typedLabel << "' to database.csv" << std::endl;
                        } else {
                            std::cout << "[Training Mode] Failed to save training instance!" << std::endl;
                        }
                    }
                } else {
                    std::cout << "[Training Mode] Empty label. Canceled." << std::endl;
                }
                isTypingLabel = false;
            } else if (key == 8 || key == 127) { // Backspace
                if (!typedLabel.empty()) {
                    typedLabel.pop_back();
                }
            } else if (key >= 32 && key <= 126) { // printable character
                typedLabel += key;
            }
        } else {
            if (key == 'q' || key == 27) { // 27 = ESC
                break;
            } else if (key == 't') {
                showThresholdInMain = !showThresholdInMain;
                std::cout << "[HUD] Toggled main window threshold view: " << (showThresholdInMain ? "ON" : "OFF") << std::endl;
            } else if (key == 'd') {
                useDynamic = !useDynamic;
                std::cout << "[HUD] Toggled dynamic thresholding: " << (useDynamic ? "ON" : "OFF") << std::endl;
            } else if (key == 'i') {
                invertThreshold = !invertThreshold;
                std::cout << "[HUD] Toggled inversion: " << (invertThreshold ? "ON" : "OFF") << std::endl;
            } else if (key == '[' || key == ',') {
                useDynamic = false; // switch to manual
                manualThreshold = std::max(0, manualThreshold - 5);
                std::cout << "[HUD] Manual Threshold decreased to: " << manualThreshold << std::endl;
            } else if (key == ']' || key == '.') {
                useDynamic = false; // switch to manual
                manualThreshold = std::min(255, manualThreshold + 5);
                std::cout << "[HUD] Manual Threshold increased to: " << manualThreshold << std::endl;
            } else if (key == 'c') {
                useCleanup = !useCleanup;
                std::cout << "[HUD] Toggled morphological cleanup: " << (useCleanup ? "ON" : "OFF") << std::endl;
            } else if (key == '-') {
                morphElementSize = std::max(0, morphElementSize - 2);
                std::cout << "[HUD] Morph Structuring Element decreased to: " << morphElementSize << std::endl;
            } else if (key == '=') {
                morphElementSize = std::min(21, morphElementSize + 2);
                std::cout << "[HUD] Morph Structuring Element increased to: " << morphElementSize << std::endl;
            } else if (key == 'n') {
                if (!useCamera && argPath.empty()) {
                    currentImgIdx = (currentImgIdx + 1) % devImages.size();
                    std::cout << "[HUD] Switched to next image: " << devImages[currentImgIdx] << std::endl;
                }
            } else if (key == 'p') {
                if (!useCamera && argPath.empty()) {
                    currentImgIdx = (currentImgIdx - 1 + devImages.size()) % devImages.size();
                    std::cout << "[HUD] Switched to previous image: " << devImages[currentImgIdx] << std::endl;
                }
            } else if (key == 'r') {
                if (features.empty()) {
                    std::cout << "[Training Mode] No regions detected in frame to train!" << std::endl;
                } else {
                    isTypingLabel = true;
                    typedLabel = "";
                    std::cout << "[Training Mode] Enter label name in the GUI window and press Enter." << std::endl;
                }
            } else if (key == 's') {
                std::time_t t = std::time(nullptr);
                char timestamp[32];
                std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&t));
                
                std::string filenameOrig = "saved_thresholds/orig_" + std::string(timestamp) + ".png";
                std::string filenameThresh = "saved_thresholds/thresh_" + std::string(timestamp) + ".png";
                std::string filenameClean = "saved_thresholds/clean_" + std::string(timestamp) + ".png";
                std::string filenameRegions = "saved_thresholds/regions_" + std::string(timestamp) + ".png";
                std::string filenameFeatures = "saved_thresholds/features_" + std::string(timestamp) + ".png";
                
                cv::Mat featSaveImg = frame.clone();
                std::vector<RegionFeatures> saveFeatures = computeRegionFeatures(labelImg, numRegions);
                drawRegionFeatures(featSaveImg, saveFeatures);

                cv::imwrite(filenameOrig, frame);
                cv::imwrite(filenameThresh, binary);
                cv::imwrite(filenameClean, cleaned);
                cv::imwrite(filenameRegions, coloredRegions);
                cv::imwrite(filenameFeatures, featSaveImg);
                std::cout << "[HUD] Saved original, raw thresholded, cleaned, region map, and features to saved_thresholds/" << std::endl;
            }
        }
    }

    if (useCamera) {
        cap.release();
    }
    cv::destroyAllWindows();
    std::cout << "[System] Application shutdown clean." << std::endl;
    return 0;
}
