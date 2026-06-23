#include "features.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

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

    // Initialize OBB extents for all regions
    std::vector<double> minE1(numRegions + 1, 1e9);
    std::vector<double> maxE1(numRegions + 1, -1e9);
    std::vector<double> minE2(numRegions + 1, 1e9);
    std::vector<double> maxE2(numRegions + 1, -1e9);

    // Compute centroids and orientations first to enable projections
    std::vector<double> cx(numRegions + 1, 0.0);
    std::vector<double> cy(numRegions + 1, 0.0);
    std::vector<double> theta(numRegions + 1, 0.0);
    for (int label = 1; label <= numRegions; ++label) {
        if (m00[label] > 0.0) {
            cx[label] = m10[label] / m00[label];
            cy[label] = m01[label] / m00[label];
            double mu20 = m20[label] - cx[label] * m10[label];
            double mu02 = m02[label] - cy[label] * m01[label];
            double mu11 = m11[label] - cx[label] * m01[label];
            theta[label] = 0.5 * std::atan2(2.0 * mu11, mu20 - mu02);
        }
    }

    // Pass 2: Project each pixel to find min/max along primary/secondary axes
    for (int r = 0; r < labelImg.rows; ++r) {
        const int *ptr = labelImg.ptr<int>(r);
        for (int c = 0; c < labelImg.cols; ++c) {
            int label = ptr[c];
            if (label > 0 && label <= numRegions && m00[label] > 0.0) {
                double dx = c - cx[label];
                double dy = r - cy[label];
                double cosT = std::cos(theta[label]);
                double sinT = std::sin(theta[label]);
                
                double proj1 = dx * cosT + dy * sinT;
                double proj2 = -dx * sinT + dy * cosT;
                
                if (proj1 < minE1[label]) minE1[label] = proj1;
                if (proj1 > maxE1[label]) maxE1[label] = proj1;
                if (proj2 < minE2[label]) minE2[label] = proj2;
                if (proj2 > maxE2[label]) maxE2[label] = proj2;
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
        rf.centroid.x = cx[label];
        rf.centroid.y = cy[label];

        // Central moments
        rf.mu20 = m20[label] - rf.centroid.x * m10[label];
        rf.mu02 = m02[label] - rf.centroid.y * m01[label];
        rf.mu11 = m11[label] - rf.centroid.x * m01[label];

        rf.orientation = theta[label];

        // Elongation
        double diff = rf.mu20 - rf.mu02;
        double term = std::sqrt(diff * diff + 4.0 * rf.mu11 * rf.mu11);
        double lambda1 = (rf.mu20 + rf.mu02 + term) / 2.0;
        double lambda2 = (rf.mu20 + rf.mu02 - term) / 2.0;

        if (std::abs(lambda2) < 1e-6) {
            rf.elongation = 1000.0;
        } else {
            rf.elongation = lambda1 / lambda2;
        }

        // Normalized central moments (scale invariant)
        double areaSq = rf.area * rf.area;
        double eta20 = rf.mu20 / areaSq;
        double eta02 = rf.mu02 / areaSq;
        double eta11 = rf.mu11 / areaSq;

        // Hu moments (rotation invariant)
        double h1 = eta20 + eta02;
        double diffEta = eta20 - eta02;
        double h2 = diffEta * diffEta + 4.0 * eta11 * eta11;

        // Build feature vector
        rf.featureVec = { rf.elongation, h1, h2 };

        // Save OBB bounds
        rf.minE1 = minE1[label];
        rf.maxE1 = maxE1[label];
        rf.minE2 = minE2[label];
        rf.maxE2 = maxE2[label];

        rf.className = "";
        rf.classDist = 0.0;

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

        // Draw Oriented Bounding Box (OBB) as green lines
        double cosT = std::cos(rf.orientation);
        double sinT = std::sin(rf.orientation);
        
        auto getOBBCorner = [&](double e1, double e2) {
            double rx = rf.centroid.x + e1 * cosT - e2 * sinT;
            double ry = rf.centroid.y + e1 * sinT + e2 * cosT;
            return cv::Point(static_cast<int>(std::round(rx)), static_cast<int>(std::round(ry)));
        };

        cv::Point pt1 = getOBBCorner(rf.maxE1, rf.maxE2);
        cv::Point pt2 = getOBBCorner(rf.minE1, rf.maxE2);
        cv::Point pt3 = getOBBCorner(rf.minE1, rf.minE2);
        cv::Point pt4 = getOBBCorner(rf.maxE1, rf.minE2);

        cv::line(dst, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        cv::line(dst, pt2, pt3, cv::Scalar(0, 255, 0), 2);
        cv::line(dst, pt3, pt4, cv::Scalar(0, 255, 0), 2);
        cv::line(dst, pt4, pt1, cv::Scalar(0, 255, 0), 2);

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

        // Draw classification result if present
        if (!rf.className.empty()) {
            std::string classText = rf.className + " (" + cv::format("%.2f", rf.classDist) + ")";
            cv::Point classTextPos = textPos + cv::Point(0, 45);
            cv::putText(dst, classText, classTextPos + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(dst, classText, classTextPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(100, 255, 100), 1, cv::LINE_AA);
        }
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

bool saveEmbeddingInstance(const std::string &dbPath, const std::string &label, const cv::Mat &embedding) {
    std::filesystem::path p(dbPath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream dbFile(dbPath, std::ios_base::app);
    if (!dbFile.is_open()) {
        std::cerr << "[Error] Could not open embedding database file: " << dbPath << std::endl;
        return false;
    }
    dbFile << label;
    int nElements = embedding.rows * embedding.cols;
    const float *ptr = embedding.ptr<float>(0);
    for (int i = 0; i < nElements; ++i) {
        dbFile << "," << ptr[i];
    }
    dbFile << "\n";
    return true;
}

std::vector<EmbeddingInstance> loadEmbeddingDatabase(const std::string &dbPath) {
    std::vector<EmbeddingInstance> db;
    std::ifstream dbFile(dbPath);
    if (!dbFile.is_open()) {
        std::cout << "[Info] Embedding database not found or empty: " << dbPath << std::endl;
        return db;
    }
    std::string line;
    while (std::getline(dbFile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string label;
        if (!std::getline(ss, label, ',')) continue;

        EmbeddingInstance inst;
        inst.label = label;
        std::string valStr;
        while (std::getline(ss, valStr, ',')) {
            try {
                inst.embedding.push_back(std::stof(valStr));
            } catch (...) {
                // Ignore conversion errors
            }
        }
        db.push_back(inst);
    }
    return db;
}

double computeCosineDistance(const cv::Mat &embA, const std::vector<float> &embB) {
    if (embA.empty() || embB.empty()) {
        return 1.0;
    }
    int nElements = embA.rows * embA.cols;
    if (nElements != static_cast<int>(embB.size())) {
        std::cerr << "[Warning] Embedding sizes mismatch: " << nElements << " vs " << embB.size() << std::endl;
        return 1.0;
    }

    const float *ptrA = embA.ptr<float>(0);
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < nElements; ++i) {
        float valA = ptrA[i];
        float valB = embB[i];
        dot += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }

    if (normA <= 0.0 || normB <= 0.0) {
        return 1.0;
    }

    double cosSim = dot / (std::sqrt(normA) * std::sqrt(normB));
    cosSim = std::max(-1.0, std::min(1.0, cosSim));
    return 1.0 - cosSim;
}
