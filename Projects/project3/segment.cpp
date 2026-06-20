#include "segment.h"
#include <vector>
#include <iostream>

int segmentRegions(const cv::Mat &src, cv::Mat &labelImg, int minArea) {
    if (src.empty()) {
        labelImg = cv::Mat();
        return 0;
    }

    // Initialize label image to 0 (background)
    labelImg = cv::Mat::zeros(src.size(), CV_32SC1);
    int labelCount = 0;

    // Scan the binary image
    for (int r = 0; r < src.rows; ++r) {
        const uchar *srcRow = src.ptr<uchar>(r);
        int *lblRow = labelImg.ptr<int>(r);

        for (int c = 0; c < src.cols; ++c) {
            // Find an unlabeled foreground pixel
            if (srcRow[c] == 255 && lblRow[c] == 0) {
                labelCount++;
                
                // Initialize BFS queue for region growing
                std::vector<cv::Point> queue;
                queue.push_back(cv::Point(c, r));
                lblRow[c] = labelCount;
                
                size_t head = 0;
                while (head < queue.size()) {
                    cv::Point p = queue[head++];
                    
                    // Scan 8-connected neighbors
                    for (int dy = -1; dy <= 1; ++dy) {
                        int ny = p.y + dy;
                        if (ny < 0 || ny >= src.rows) continue;
                        
                        int *nyLblRow = labelImg.ptr<int>(ny);
                        const uchar *nySrcRow = src.ptr<uchar>(ny);
                        
                        for (int dx = -1; dx <= 1; ++dx) {
                            int nx = p.x + dx;
                            if (nx < 0 || nx >= src.cols) continue;
                            
                            // If neighbor is foreground and unlabeled
                            if (nySrcRow[nx] == 255 && nyLblRow[nx] == 0) {
                                nyLblRow[nx] = labelCount;
                                queue.push_back(cv::Point(nx, ny));
                            }
                        }
                    }
                }
                
                // Filter region by area
                if (static_cast<int>(queue.size()) < minArea) {
                    // Reset discarded pixels to 0
                    for (const auto &pt : queue) {
                        labelImg.at<int>(pt.y, pt.x) = 0;
                    }
                    // Reuse this label ID for the next valid component
                    labelCount--;
                }
            }
        }
    }
    
    return labelCount;
}

void colorizeRegions(const cv::Mat &labelImg, cv::Mat &dst, int numRegions) {
    if (labelImg.empty()) {
        dst = cv::Mat();
        return;
    }

    dst = cv::Mat::zeros(labelImg.size(), CV_8UC3);
    
    if (numRegions <= 0) {
        return; // Return all black image
    }

    // Generate distinct, high-contrast BGR colors using HSV color space
    std::vector<cv::Vec3b> colors(numRegions + 1);
    colors[0] = cv::Vec3b(0, 0, 0); // background is black
    
    for (int i = 1; i <= numRegions; ++i) {
        // Deterministically distribute hues across HSV spectrum (0-179)
        int h = (i * 131) % 180;
        int s = 210 + (i * 17) % 46; // High saturation
        int v = 200 + (i * 23) % 56; // High brightness
        
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        colors[i] = bgr.at<cv::Vec3b>(0, 0);
    }

    // Assign colors to pixels based on their label ID
    for (int r = 0; r < labelImg.rows; ++r) {
        const int *lblPtr = labelImg.ptr<int>(r);
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(r);
        
        for (int c = 0; c < labelImg.cols; ++c) {
            int lblVal = lblPtr[c];
            if (lblVal > 0 && lblVal <= numRegions) {
                dstPtr[c] = colors[lblVal];
            } else {
                dstPtr[c] = colors[0];
            }
        }
    }
}
