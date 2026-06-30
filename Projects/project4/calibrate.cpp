#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

// Constants for our checkerboard
const int BOARD_COLS = 9; // Internal corners horizontally
const int BOARD_ROWS = 6; // Internal corners vertically
const cv::Size BOARD_SIZE(BOARD_COLS, BOARD_ROWS);

// Generate 3D world coordinates for the chessboard corners
std::vector<cv::Vec3f> get_chessboard_world_points() {
    std::vector<cv::Vec3f> point_set;
    for (int r = 0; r < BOARD_ROWS; r++) {
        for (int c = 0; c < BOARD_COLS; c++) {
            // Origin is top-left, y goes negative so Z points toward viewer
            point_set.push_back(cv::Vec3f(static_cast<float>(c), -static_cast<float>(r), 0.0f));
        }
    }
    return point_set;
}

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

    cv::VideoCapture cap;
    bool is_webcam = true;
    std::vector<std::string> image_paths;
    int current_img_idx = 0;

    if (!input_source.empty()) {
        if (std::filesystem::is_directory(input_source)) {
            // Scan directory for images
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
            std::cout << "Loaded " << image_paths.size() << " images for calibration from directory." << std::endl;
        } else {
            // Treat as video file
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
        // Open default camera
        if (!cap.open(0)) {
            std::cerr << "Error: Could not open webcam." << std::endl;
            return -1;
        }
        std::cout << "Opened default webcam." << std::endl;
    }

    if (!headless) {
        cv::namedWindow("Calibration Target Detection", cv::WINDOW_AUTOSIZE);
    }

    // Lists for calibration data
    std::vector<cv::Vec3f> point_set = get_chessboard_world_points();
    std::vector<std::vector<cv::Vec3f>> point_list;
    std::vector<std::vector<cv::Point2f>> corner_list;
    std::vector<cv::Mat> saved_frames; // Store actual images used

    // Camera matrix and distortion parameters
    cv::Mat camera_matrix;
    std::vector<double> dist_coeffs(5, 0.0); // 5 radial/tangential distortion parameters
    bool calibrated = false;

    if (!headless) {
        // Interactive guidance
        std::cout << "\n--- Calibration Controls ---\n"
                  << "Press 's' to save the current frame for calibration.\n"
                  << "Press 'c' to run calibration (requires at least 5 saved frames).\n"
                  << "Press 'w' to write camera parameters to 'calibration_params.xml'.\n"
                  << "Press 'q' to quit.\n"
                  << "----------------------------\n" << std::endl;
    }

    cv::Mat frame;
    cv::Mat last_valid_frame;
    std::vector<cv::Point2f> last_valid_corners;
    bool target_detected = false;

    while (true) {
        if (is_webcam || cap.isOpened()) {
            cap >> frame;
            if (frame.empty()) {
                if (is_webcam) {
                    std::cerr << "Empty frame from webcam." << std::endl;
                    break;
                } else {
                    std::cout << "Reached end of video file." << std::endl;
                    break;
                }
            }
        } else {
            // Load from image list
            if (current_img_idx >= static_cast<int>(image_paths.size())) {
                std::cout << "Finished processing all images in directory." << std::endl;
                break;
            }
            frame = cv::imread(image_paths[current_img_idx]);
            if (frame.empty()) {
                std::cerr << "Failed to read image: " << image_paths[current_img_idx] << std::endl;
                current_img_idx++;
                continue;
            }
            std::cout << "Processing image [" << current_img_idx + 1 << "/" << image_paths.size() 
                      << "]: " << image_paths[current_img_idx] << std::endl;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corner_set;
        // Task 2: Detect chessboard corners
        bool found = cv::findChessboardCorners(gray, BOARD_SIZE, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        cv::Mat display_frame;
        if (!headless) {
            display_frame = frame.clone();
        }

        if (found) {
            // Subpixel refinement
            cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            if (!headless) {
                // Visualization
                cv::drawChessboardCorners(display_frame, BOARD_SIZE, corner_set, found);
            }

            // Print corners info
            static int print_counter = 0;
            if (print_counter++ % 30 == 0 || !is_webcam) {
                std::cout << "Detected chessboard! Corners found: " << corner_set.size() 
                          << " | First corner: (" << corner_set[0].x << ", " << corner_set[0].y << ")" << std::endl;
            }

            // Keep track of the last successful detection
            last_valid_frame = frame.clone();
            last_valid_corners = corner_set;
            target_detected = true;

            // In headless mode, automatically save every detected checkerboard
            if (headless) {
                corner_list.push_back(corner_set);
                point_list.push_back(point_set);
                saved_frames.push_back(frame);
                std::cout << "--> Auto-saved calibration frame #" << corner_list.size() << std::endl;

                // Save raw and annotated copies for report inclusion / verification
                std::string saved_fn = "calibration_frame_" + std::to_string(corner_list.size()) + ".png";
                cv::imwrite(saved_fn, frame);
                std::cout << "Saved: " << saved_fn << std::endl;

                cv::Mat annotated_frame = frame.clone();
                cv::drawChessboardCorners(annotated_frame, BOARD_SIZE, corner_set, true);
                std::string saved_corners_fn = "calibration_frame_" + std::to_string(corner_list.size()) + "_corners.png";
                cv::imwrite(saved_corners_fn, annotated_frame);
                std::cout << "Saved annotated frame: " << saved_corners_fn << std::endl;
            }
        } else {
            target_detected = false;
        }

        if (!headless) {
            // Show status on image
            std::string status = "Saved Frames: " + std::to_string(corner_list.size());
            cv::putText(display_frame, status, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            if (calibrated) {
                cv::putText(display_frame, "Calibrated!", cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
            }

            cv::imshow("Calibration Target Detection", display_frame);

            // Keypress handling
            char key = 0;
            if (is_webcam || cap.isOpened()) {
                key = static_cast<char>(cv::waitKey(30));
            } else {
                // For static images, wait for keypress to proceed
                std::cout << "Press 's' to save this image, 'n' or any key for next image..." << std::endl;
                key = static_cast<char>(cv::waitKey(0));
            }

            if (key == 'q') {
                break;
            } else if (key == 's') {
                // Task 3: Save calibration image
                if (target_detected || (!last_valid_corners.empty())) {
                    cv::Mat frame_to_save = target_detected ? frame : last_valid_frame;
                    std::vector<cv::Point2f> corners_to_save = target_detected ? corner_set : last_valid_corners;

                    corner_list.push_back(corners_to_save);
                    point_list.push_back(point_set);
                    saved_frames.push_back(frame_to_save);

                    std::cout << "--> Saved calibration frame #" << corner_list.size() << std::endl;
                    
                    std::string saved_fn = "calibration_frame_" + std::to_string(corner_list.size()) + ".png";
                    cv::imwrite(saved_fn, frame_to_save);
                    std::cout << "Saved: " << saved_fn << std::endl;

                    // Also save an annotated version with the corners highlighted
                    cv::Mat annotated_frame = frame_to_save.clone();
                    cv::drawChessboardCorners(annotated_frame, BOARD_SIZE, corners_to_save, true);
                    std::string saved_corners_fn = "calibration_frame_" + std::to_string(corner_list.size()) + "_corners.png";
                    cv::imwrite(saved_corners_fn, annotated_frame);
                    std::cout << "Saved annotated frame: " << saved_corners_fn << std::endl;
                } else {
                    std::cout << "Warning: Chessboard target was not detected in this or any recent frame. Cannot save." << std::endl;
                }
            } else if (key == 'c') {
                // Task 4: Calibrate camera
                if (corner_list.size() < 5) {
                    std::cout << "Error: You need at least 5 saved calibration frames. Currently have: " 
                              << corner_list.size() << std::endl;
                } else {
                    std::cout << "\nStarting camera calibration with " << corner_list.size() << " frames..." << std::endl;
                    
                    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                    camera_matrix.at<double>(0, 2) = frame.cols / 2.0;
                    camera_matrix.at<double>(1, 2) = frame.rows / 2.0;

                    std::cout << "Initial camera matrix:\n" << camera_matrix << std::endl;

                    std::vector<cv::Mat> rvecs, tvecs;
                    double error = cv::calibrateCamera(
                        point_list,
                        corner_list,
                        frame.size(),
                        camera_matrix,
                        dist_coeffs,
                        rvecs,
                        tvecs,
                        cv::CALIB_FIX_ASPECT_RATIO
                    );

                    std::cout << "Calibration SUCCESSFUL!" << std::endl;
                    std::cout << "Final Re-projection Error: " << error << " pixels" << std::endl;
                    std::cout << "Calibrated camera matrix:\n" << camera_matrix << std::endl;
                    std::cout << "Calibrated distortion coefficients:\n";
                    for (double d : dist_coeffs) std::cout << d << " ";
                    std::cout << "\n" << std::endl;

                    calibrated = true;
                }
            } else if (key == 'w') {
                if (!calibrated) {
                    std::cout << "Error: Camera must be calibrated before writing parameters." << std::endl;
                } else {
                    cv::FileStorage fs("calibration_params.xml", cv::FileStorage::WRITE);
                    if (fs.isOpened()) {
                        fs << "camera_matrix" << camera_matrix;
                        fs << "distortion_coefficients" << cv::Mat(dist_coeffs);
                        fs.release();
                        std::cout << "Wrote intrinsic parameters to 'calibration_params.xml'" << std::endl;
                    } else {
                        std::cerr << "Error: Could not open 'calibration_params.xml' for writing." << std::endl;
                    }
                }
            }
        }

        // For static images, advance to the next one
        if (!is_webcam && !cap.isOpened()) {
            current_img_idx++;
        }
    }

    // Auto-calibrate and save in headless mode
    if (headless) {
        if (corner_list.size() < 5) {
            std::cerr << "Error: Headless calibration needs at least 5 detected frames. Found: " 
                      << corner_list.size() << std::endl;
            return -1;
        }
        std::cout << "\n[Headless] Starting camera calibration with " << corner_list.size() << " frames..." << std::endl;
        
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        camera_matrix.at<double>(0, 2) = frame.cols / 2.0;
        camera_matrix.at<double>(1, 2) = frame.rows / 2.0;

        std::vector<cv::Mat> rvecs, tvecs;
        double error = cv::calibrateCamera(
            point_list,
            corner_list,
            frame.size(),
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            cv::CALIB_FIX_ASPECT_RATIO
        );

        std::cout << "Calibration SUCCESSFUL!" << std::endl;
        std::cout << "Final Re-projection Error: " << error << " pixels" << std::endl;
        std::cout << "Calibrated camera matrix:\n" << camera_matrix << std::endl;
        std::cout << "Calibrated distortion coefficients:\n";
        for (double d : dist_coeffs) std::cout << d << " ";
        std::cout << "\n" << std::endl;

        cv::FileStorage fs("calibration_params.xml", cv::FileStorage::WRITE);
        if (fs.isOpened()) {
            fs << "camera_matrix" << camera_matrix;
            fs << "distortion_coefficients" << cv::Mat(dist_coeffs);
            fs.release();
            std::cout << "Wrote intrinsic parameters to 'calibration_params.xml'" << std::endl;
        } else {
            std::cerr << "Error: Could not open 'calibration_params.xml' for writing." << std::endl;
            return -1;
        }
    }

    if (!headless) {
        cv::destroyAllWindows();
    }
    return 0;
}
