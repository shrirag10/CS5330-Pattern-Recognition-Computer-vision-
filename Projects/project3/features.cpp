#include "features.h"
#include <cmath>
#include <iostream>
#include <fstream>

std::vector<RegionFeatures> computeRegionFeatures(const cv::Mat &labelImg, int numRegions) {
    std::vector<RegionFeatures> results;
    if (labelImg.empty() || numRegions <= 0) {
        return results;
    }

    // Allocate memory for moment accumulations
    // Using double to prevent overflow in coordinate sums
    std::vector<double> m00(numRegions + 1, 0.0);
    std::vector<double> m10(numRegions + 1, 0.0);
    std::vector<double> m01(numRegions + 1, 0.0);
    std::vector<double> m20(numRegions + 1, 0.0);
    std::vector<double> m02(numRegions + 1, 0.0);
    std::vector<double> m11(numRegions + 1, 0.0);

    // Single pass accumulation
    for (int r = 0; r < labelImg.rows; ++r) {
        const int *lblPtr = labelImg.ptr<int>(r);
        for (int c = 0; c < labelImg.cols; ++c) {
            int label = lblPtr[c];
            if (label > 0 && label <= numRegions) {
                m00[label] += 1.0;
                m10[label] += c;
                m01[label] += r;
                m20[label] += static_cast<double>(c) * c;
                m02[label] += static_cast<double>(r) * r;
                m11[label] += static_cast<double>(c) * r;
            }
        }
    }

    // Compute derived features for each region
    for (int label = 1; label <= numRegions; ++label) {
        if (m00[label] <= 0.0) {
            continue; // Skipped / pruned region
        }

        RegionFeatures rf;
        rf.labelId = label;
        rf.area = m00[label];

        // 1. Centroid coordinates
        rf.centroid.x = m10[label] / m00[label];
        rf.centroid.y = m01[label] / m00[label];

        // 2. Central moments (translation invariant)
        rf.mu20 = m20[label] - rf.centroid.x * m10[label];
        rf.mu02 = m02[label] - rf.centroid.y * m01[label];
        rf.mu11 = m11[label] - rf.centroid.x * m01[label];

        // 3. Orientation (angle of axis of least moment of inertia)
        // Range is -pi/2 to pi/2
        rf.orientation = 0.5 * std::atan2(2.0 * rf.mu11, rf.mu20 - rf.mu02);

        // 4. Elongation via eigenvalues of the covariance tensor
        // Covariance matrix elements: [mu20/m00, mu11/m00; mu11/m00, mu02/m00]
        // The eigenvalues represent the principal moments of inertia
        double diff = rf.mu20 - rf.mu02;
        double term = std::sqrt(diff * diff + 4.0 * rf.mu11 * rf.mu11);
        double lambda1 = (rf.mu20 + rf.mu02 + term) / 2.0;
        double lambda2 = (rf.mu20 + rf.mu02 - term) / 2.0;

        if (std::abs(lambda2) < 1e-6) {
            rf.elongation = 1000.0; // Clamped large value for highly elongated objects
        } else {
            rf.elongation = lambda1 / lambda2;
        }

        // 5. Normalized central moments (scale invariant)
        double areaSq = rf.area * rf.area;
        double eta20 = rf.mu20 / areaSq;
        double eta02 = rf.mu02 / areaSq;
        double eta11 = rf.mu11 / areaSq;

        // 6. Hu moments (rotation invariant)
        double h1 = eta20 + eta02;
        double diffEta = eta20 - eta02;
        double h2 = diffEta * diffEta + 4.0 * eta11 * eta11;

        // Build feature vector
        rf.featureVec = { rf.elongation, h1, h2 };

        results.push_back(rf);
    }

    return results;
}

void drawRegionFeatures(cv::Mat &dst, const std::vector<RegionFeatures> &features) {
    if (dst.empty() || features.empty()) {
        return;
    }

    for (const auto &rf : features) {
        // Draw Centroid as a red dot and crosshair
        cv::Point centroidPt(static_cast<int>(std::round(rf.centroid.x)), 
                             static_cast<int>(std::round(rf.centroid.y)));
        cv::circle(dst, centroidPt, 4, cv::Scalar(0, 0, 255), -1); // Red dot
        
        int crossSize = 8;
        cv::line(dst, cv::Point(centroidPt.x - crossSize, centroidPt.y), 
                      cv::Point(centroidPt.x + crossSize, centroidPt.y), cv::Scalar(0, 0, 255), 2);
        cv::line(dst, cv::Point(centroidPt.x, centroidPt.y - crossSize), 
                      cv::Point(centroidPt.x, centroidPt.y + crossSize), cv::Scalar(0, 0, 255), 2);

        // Draw Orientation Axis as a blue line
        // Length of the axis is proportional to the square root of region area
        double axisLen = 1.0 * std::sqrt(rf.area);
        double dx = axisLen * std::cos(rf.orientation);
        double dy = axisLen * std::sin(rf.orientation);
        cv::Point endPt1(static_cast<int>(std::round(rf.centroid.x + dx)), 
                         static_cast<int>(std::round(rf.centroid.y + dy)));
        cv::Point endPt2(static_cast<int>(std::round(rf.centroid.x - dx)), 
                         static_cast<int>(std::round(rf.centroid.y - dy)));
        cv::line(dst, endPt1, endPt2, cv::Scalar(255, 0, 0), 2); // Blue axis line

        // Draw text info: Label ID, Elongation, and Orientation (in degrees)
        std::string labelText = "ID: " + std::to_string(rf.labelId);
        std::string elonText  = "Elong: " + cv::format("%.2f", rf.elongation);
        std::string angleText = "Angle: " + cv::format("%.1f", rf.orientation * 180.0 / CV_PI) + " deg";

        cv::Point textPos(centroidPt.x + 12, centroidPt.y - 12);
        
        // Render text with subtle black shadows for readability
        cv::putText(dst, labelText, textPos + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(dst, labelText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        cv::putText(dst, elonText, textPos + cv::Point(1, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(dst, elonText, textPos + cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        cv::putText(dst, angleText, textPos + cv::Point(1, 31), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(dst, angleText, textPos + cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}

bool saveTrainingInstance(const std::string &dbPath, const std::string &label, const std::vector<double> &features) {
    std::ofstream dbFile(dbPath, std::ios_base::app);
    if (!dbFile.is_open()) {
        std::cerr << "[Error] Could not open database file: " << dbPath << std::endl;
        return false;
    }
    dbFile << label;
    for (double val : features) {
        dbFile << "," << val;
    }
    dbFile << "\n";
    return true;
}
