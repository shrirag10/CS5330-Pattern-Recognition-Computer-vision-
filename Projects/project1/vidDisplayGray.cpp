#include <iostream>
#include <ctime>
#include <filesystem>
#include <opencv2/opencv.hpp>


int main(int argc, char *argv[]) {
    cv::VideoCapture cap(0);  // 0 = default webcam
    if (!cap.isOpened()) {
        std::cout << "Could not open webcam" << std::endl;
        return -1;
    }

    std::cout << "Webcam opened successfully" << std::endl;
    std::cout << "Press 's' to save frame, 'q' to quit" << std::endl;

    std::filesystem::create_directories("task2");

    cv::Mat frame;
    int grayToggle = 0;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // cv::imshow("Webcam", frame);

        int key = cv::waitKey(1);
        if(key == 'g') {
            if (grayToggle == 0) {
                grayToggle = 1;
            } else {
                grayToggle = 0;
            }
        }

        cv::Mat newFrame = frame.clone();
        if (grayToggle == 1) {
            cv::cvtColor(frame, newFrame, cv::COLOR_BGR2GRAY);
        }
        cv::imshow("Webcam",  newFrame);

        if (key == 'q') break;
        if (key == 's') {
            std::time_t t = std::time(nullptr);
            char timestamp[32];
            std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&t));
            std::string filename = std::string("task2/frame_") + timestamp + ".png";
            cv::imwrite(filename, newFrame);
            std::cout << "Saved: " << filename << std::endl;
        }

    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}