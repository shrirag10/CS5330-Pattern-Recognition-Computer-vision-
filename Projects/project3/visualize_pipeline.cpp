/*
  visualize_pipeline.cpp
  Runs one image through the full recognition pipeline and saves a PNG at each stage.

  Usage:
    ./build/visualize_pipeline --input <image> [--output <dir>] [--prefix <name>]

  Output files (in <dir>, default report_imgs/):
    <prefix>_1_original.png
    <prefix>_2_threshold.png
    <prefix>_3_cleanup.png
    <prefix>_4_regions.png
    <prefix>_5_features.png
*/

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "features.h"

namespace fs = std::filesystem;

static cv::Mat scaleDown(const cv::Mat &img, int maxDim = 800) {
    if (img.cols <= maxDim && img.rows <= maxDim) return img.clone();
    double scale = (double)maxDim / std::max(img.cols, img.rows);
    cv::Mat out;
    cv::resize(img, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

int main(int argc, char *argv[]) {
    std::string inputPath;
    std::string outputDir = "report_imgs";
    std::string prefix    = "pipeline";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input"  && i + 1 < argc) inputPath  = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputDir = argv[++i];
        else if (arg == "--prefix" && i + 1 < argc) prefix    = argv[++i];
    }

    if (inputPath.empty()) {
        std::cerr << "Usage: visualize_pipeline --input <image> [--output <dir>] [--prefix <name>]\n";
        return 1;
    }

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "[Error] Could not read: " << inputPath << "\n";
        return 1;
    }

    fs::create_directories(outputDir);
    img = scaleDown(img, 800);

    auto save = [&](const std::string &tag, const cv::Mat &frame) {
        std::string path = outputDir + "/" + prefix + tag;
        cv::imwrite(path, frame);
        std::cout << "[Saved] " << path << "\n";
    };

    // --- Stage 1: Original ---
    save("_1_original.png", img);

    // --- Stage 2: Threshold ---
    int thresh = computeDynamicThresholdISODATA(img, 4);
    cv::Mat binary;
    thresholdBinary(img, binary, thresh, true);
    cv::Mat threshVis;
    cv::cvtColor(binary, threshVis, cv::COLOR_GRAY2BGR);
    save("_2_threshold.png", threshVis);
    std::cout << "[Info]  ISODATA threshold = " << thresh << "\n";

    // --- Stage 3: Morphological cleanup ---
    cv::Mat cleaned;
    cleanupBinary(binary, cleaned, 5, 3);
    cv::Mat cleanVis;
    cv::cvtColor(cleaned, cleanVis, cv::COLOR_GRAY2BGR);
    save("_3_cleanup.png", cleanVis);

    // --- Stage 4: Segmentation (colorized regions) ---
    cv::Mat labelImg;
    int numRegions = segmentRegions(cleaned, labelImg, 500);
    cv::Mat regionVis;
    colorizeRegions(labelImg, regionVis, numRegions);
    save("_4_regions.png", regionVis);
    std::cout << "[Info]  Regions found = " << numRegions << "\n";

    if (numRegions == 0) {
        std::cerr << "[Warn]  No regions found — skipping feature overlay.\n";
        return 0;
    }

    // --- Stage 5: Feature overlay (OBB + axis + centroid) ---
    std::vector<RegionFeatures> feats = computeRegionFeatures(labelImg, numRegions);

    // pick largest region
    RegionFeatures *largest = &feats[0];
    for (auto &rf : feats)
        if (rf.area > largest->area) largest = &rf;

    cv::Mat featVis = img.clone();

    for (const auto &rf : feats) {
        int cx = (int)rf.centroid.x;
        int cy = (int)rf.centroid.y;

        // centroid crosshair
        cv::drawMarker(featVis, cv::Point(cx, cy), cv::Scalar(0, 255, 255),
                       cv::MARKER_CROSS, 16, 2);

        // primary axis line (eigenvector direction)
        double axisLen = 60.0;
        int dx = (int)(axisLen * std::cos(rf.orientation));
        int dy = (int)(axisLen * std::sin(rf.orientation));
        cv::arrowedLine(featVis, cv::Point(cx - dx, cy - dy),
                        cv::Point(cx + dx, cy + dy),
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.2);

        // OBB corners via primary/secondary axis projections
        double cosT = std::cos(rf.orientation);
        double sinT = std::sin(rf.orientation);
        double e1s[2] = {rf.minE1, rf.maxE1};
        double e2s[2] = {rf.minE2, rf.maxE2};
        std::vector<cv::Point> corners;
        for (double e1 : e1s)
            for (double e2 : e2s)
                corners.push_back(cv::Point(
                    cx + (int)(e1 * cosT - e2 * sinT),
                    cy + (int)(e1 * sinT + e2 * cosT)));
        // reorder: 00 01 11 10 for a proper quadrilateral
        std::vector<cv::Point> quad = {corners[0], corners[1], corners[3], corners[2]};
        cv::polylines(featVis, quad, true, cv::Scalar(255, 80, 0), 2, cv::LINE_AA);

        // label: elongation
        char buf[64];
        snprintf(buf, sizeof(buf), "elong=%.2f", rf.elongation);
        cv::putText(featVis, buf, cv::Point(cx + 8, cy - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }

    save("_5_features.png", featVis);

    // --- Stage 6: Pipeline summary strip (1x5 thumbnails side by side) ---
    const int thumbH = 200;
    std::vector<std::string> stageTags = {
        "_1_original.png", "_2_threshold.png", "_3_cleanup.png",
        "_4_regions.png",  "_5_features.png"
    };
    std::vector<std::string> stageLabels = {
        "1 Original", "2 Threshold", "3 Cleanup", "4 Regions", "5 Features"
    };

    std::vector<cv::Mat> thumbs;
    for (size_t i = 0; i < stageTags.size(); i++) {
        cv::Mat stage = cv::imread(outputDir + "/" + prefix + stageTags[i]);
        if (stage.empty()) continue;
        double scale = (double)thumbH / stage.rows;
        cv::Mat thumb;
        cv::resize(stage, thumb, cv::Size(), scale, scale, cv::INTER_AREA);
        // add label banner
        cv::copyMakeBorder(thumb, thumb, 24, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(30, 30, 30));
        cv::putText(thumb, stageLabels[i], cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
        thumbs.push_back(thumb);
    }
    if (!thumbs.empty()) {
        cv::Mat strip;
        cv::hconcat(thumbs, strip);
        save("_0_strip.png", strip);
    }

    std::cout << "[Done]  All pipeline stages saved to " << outputDir << "/\n";
    return 0;
}
