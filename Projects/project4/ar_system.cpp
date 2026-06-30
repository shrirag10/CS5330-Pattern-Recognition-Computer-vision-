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

// Define a structure for 3D lines in our virtual object
struct Line3D {
    cv::Vec3f pt1;
    cv::Vec3f pt2;
    cv::Scalar color;
    int thickness;
};

// Generate 3D lines of a virtual asymmetrical house floating above the board
std::vector<Line3D> get_virtual_house() {
    std::vector<Line3D> lines;

    // Base square corners (floating at Z = 1.0)
    cv::Vec3f b0(3.0f, -1.0f, 1.0f);
    cv::Vec3f b1(5.0f, -1.0f, 1.0f);
    cv::Vec3f b2(5.0f, -3.0f, 1.0f);
    cv::Vec3f b3(3.0f, -3.0f, 1.0f);

    // Top square corners (floating at Z = 3.0)
    cv::Vec3f t0(3.0f, -1.0f, 3.0f);
    cv::Vec3f t1(5.0f, -1.0f, 3.0f);
    cv::Vec3f t2(5.0f, -3.0f, 3.0f);
    cv::Vec3f t3(3.0f, -3.0f, 3.0f);

    // Off-center roof apex (Z = 4.5, shifted slightly for asymmetry)
    cv::Vec3f apex(4.3f, -2.2f, 4.5f);

    // Chimney corners (small vertical box on the roof)
    cv::Vec3f c0(3.2f, -1.5f, 3.2f);
    cv::Vec3f c1(3.6f, -1.5f, 3.6f);
    cv::Vec3f c2(3.6f, -1.5f, 4.0f); // Top of chimney

    // Colors
    cv::Scalar color_base(0, 0, 255);      // Red
    cv::Scalar color_pillars(0, 255, 0);   // Green
    cv::Scalar color_top(255, 0, 0);       // Blue
    cv::Scalar color_roof(255, 255, 0);    // Cyan
    cv::Scalar color_chimney(255, 0, 255);  // Magenta

    // 1. Base square lines
    lines.push_back({b0, b1, color_base, 3});
    lines.push_back({b1, b2, color_base, 3});
    lines.push_back({b2, b3, color_base, 3});
    lines.push_back({b3, b0, color_base, 3});

    // 2. Top square lines
    lines.push_back({t0, t1, color_top, 3});
    lines.push_back({t1, t2, color_top, 3});
    lines.push_back({t2, t3, color_top, 3});
    lines.push_back({t3, t0, color_top, 3});

    // 3. Vertical pillar lines (connecting base to top)
    lines.push_back({b0, t0, color_pillars, 2});
    lines.push_back({b1, t1, color_pillars, 2});
    lines.push_back({b2, t2, color_pillars, 2});
    lines.push_back({b3, t3, color_pillars, 2});

    // 4. Roof lines (connecting top corners to apex)
    lines.push_back({t0, apex, color_roof, 2});
    lines.push_back({t1, apex, color_roof, 2});
    lines.push_back({t2, apex, color_roof, 2});
    lines.push_back({t3, apex, color_roof, 2});

    // 5. Chimney lines
    lines.push_back({c0, c1, color_chimney, 2});
    lines.push_back({c1, c2, color_chimney, 2});

    return lines;
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

    // Load calibration parameters
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::FileStorage fs("calibration_params.xml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Could not open 'calibration_params.xml'. Please run calibration first." << std::endl;
        return -1;
    }
    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;
    fs.release();

    std::cout << "Loaded camera matrix:\n" << camera_matrix << std::endl;
    std::cout << "Loaded distortion coefficients:\n" << dist_coeffs << std::endl;

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
            std::cout << "Loaded " << image_paths.size() << " images for AR rendering." << std::endl;
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

    if (!headless) {
        cv::namedWindow("Augmented Reality Overlay", cv::WINDOW_AUTOSIZE);
    } else {
        std::filesystem::create_directories("data/ar_output");
    }

    std::vector<cv::Vec3f> world_points = get_chessboard_world_points();
    std::vector<Line3D> virtual_object = get_virtual_house();

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
            std::cout << "Rendering AR for image: " << image_paths[current_img_idx] << std::endl;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corner_set;
        bool found = cv::findChessboardCorners(gray, BOARD_SIZE, corner_set,
            cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        cv::Mat annotated_frame = frame.clone();

        if (found) {
            // Subpixel refinement
            cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            // Task 5: Solve PNP for pose
            cv::Mat rvec, tvec;
            bool success = cv::solvePnP(world_points, corner_set, camera_matrix, dist_coeffs, rvec, tvec);

            if (success) {
                // Print translation and rotation vectors
                std::cout << "Pose - Translation tvec:\n" << tvec.t() 
                          << "\nPose - Rotation rvec:\n" << rvec.t() << std::endl;

                // Task 5: Project and Draw 3D Axes
                // Axis points in world coordinates: Origin (0,0,0), X(3,0,0), Y(0,-3,0), Z(0,0,3)
                std::vector<cv::Vec3f> axis_points = {
                    cv::Vec3f(0.0f, 0.0f, 0.0f),
                    cv::Vec3f(3.0f, 0.0f, 0.0f),
                    cv::Vec3f(0.0f, -3.0f, 0.0f),
                    cv::Vec3f(0.0f, 0.0f, 3.0f)
                };
                std::vector<cv::Point2f> projected_axes;
                cv::projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs, projected_axes);

                // Draw X-axis (Red)
                cv::line(annotated_frame, projected_axes[0], projected_axes[1], cv::Scalar(0, 0, 255), 3);
                // Draw Y-axis (Green)
                cv::line(annotated_frame, projected_axes[0], projected_axes[2], cv::Scalar(0, 255, 0), 3);
                // Draw Z-axis (Blue)
                cv::line(annotated_frame, projected_axes[0], projected_axes[3], cv::Scalar(255, 0, 0), 3);

                // Add text labels
                cv::putText(annotated_frame, "X", projected_axes[1], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                cv::putText(annotated_frame, "Y", projected_axes[2], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                cv::putText(annotated_frame, "Z", projected_axes[3], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);

                // Task 6: Project and Draw the Virtual Object
                for (const auto &line : virtual_object) {
                    std::vector<cv::Vec3f> line_pts = {line.pt1, line.pt2};
                    std::vector<cv::Point2f> proj_pts;
                    cv::projectPoints(line_pts, rvec, tvec, camera_matrix, dist_coeffs, proj_pts);

                    // Draw line
                    cv::line(annotated_frame, proj_pts[0], proj_pts[1], line.color, line.thickness);
                }
            }
        } else {
            if (!headless) {
                cv::putText(annotated_frame, "Target Not Found", cv::Point(20, 50), 
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }
        }

        if (!headless) {
            cv::imshow("Augmented Reality Overlay", annotated_frame);
            char key = static_cast<char>(cv::waitKey(is_webcam ? 30 : 0));
            if (key == 'q') {
                break;
            }
        } else {
            // Save output frame
            std::string out_fn = "data/ar_output/ar_frame_" + std::to_string(current_img_idx + 1) + ".png";
            cv::imwrite(out_fn, annotated_frame);
            std::cout << "Saved: " << out_fn << std::endl;
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
