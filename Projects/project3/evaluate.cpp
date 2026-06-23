#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "features.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    std::string datasetDir = "train_data";
    std::string dbPath = "database.csv";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) datasetDir = argv[++i];
        else if (arg == "--db" && i + 1 < argc) dbPath = argv[++i];
    }

    std::vector<TrainingEntry> db = loadDatabase(dbPath);
    if (db.empty()) {
        std::cerr << "[Error] Could not load database from: " << dbPath << std::endl;
        return 1;
    }

    // collect all known labels in sorted order
    std::vector<std::string> labels;
    for (const auto &e : db) {
        bool found = false;
        for (const auto &l : labels) if (l == e.label) { found = true; break; }
        if (!found) labels.push_back(e.label);
    }
    std::sort(labels.begin(), labels.end());
    int N = (int)labels.size();

    auto labelIndex = [&](const std::string &l) {
        for (int i = 0; i < N; i++) if (labels[i] == l) return i;
        return -1;
    };

    // confusion matrix: matrix[true][predicted]
    std::vector<std::vector<int>> matrix(N, std::vector<int>(N, 0));
    int total = 0, correct = 0, skipped = 0;

    for (const auto &classEntry : fs::directory_iterator(datasetDir)) {
        if (!classEntry.is_directory()) continue;
        std::string trueLabel = classEntry.path().filename().string();
        int trueIdx = labelIndex(trueLabel);
        if (trueIdx < 0) {
            std::cerr << "[Skip] Label not in DB: " << trueLabel << std::endl;
            continue;
        }

        for (const auto &imgEntry : fs::directory_iterator(classEntry.path())) {
            std::string ext = imgEntry.path().extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

            cv::Mat img = cv::imread(imgEntry.path().string());
            if (img.empty()) { skipped++; continue; }

            int thresh = computeDynamicThresholdISODATA(img, 4);
            cv::Mat binary, cleaned, labelImg;
            thresholdBinary(img, binary, thresh, true);
            cleanupBinary(binary, cleaned, 5, 3);
            int numRegions = segmentRegions(cleaned, labelImg, 500);

            if (numRegions == 0) { skipped++; continue; }

            std::vector<RegionFeatures> features = computeRegionFeatures(labelImg, numRegions);
            RegionFeatures largest = features[0];
            for (const auto &rf : features)
                if (rf.area > largest.area) largest = rf;

            std::string predicted = classifyFeatures(largest.featureVec, db);
            int predIdx = labelIndex(predicted);

            if (predIdx >= 0) matrix[trueIdx][predIdx]++;
            if (predicted == trueLabel) correct++;
            total++;

            std::cout << "[" << (predicted == trueLabel ? "OK" : "MISS") << "] "
                      << trueLabel << " -> " << predicted
                      << " (" << imgEntry.path().filename().string() << ")" << std::endl;
        }
    }

    // print confusion matrix
    std::cout << "\nConfusion Matrix (rows=true, cols=predicted):\n\n";
    int col_w = 12;
    std::cout << std::string(col_w, ' ');
    for (const auto &l : labels)
        std::cout << std::setw(col_w) << l;
    std::cout << "\n" << std::string(col_w * (N + 1), '-') << "\n";
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(col_w) << labels[i];
        for (int j = 0; j < N; j++)
            std::cout << std::setw(col_w) << matrix[i][j];
        std::cout << "\n";
    }

    double accuracy = total > 0 ? (100.0 * correct / total) : 0.0;
    std::cout << "\nAccuracy: " << correct << "/" << total
              << " (" << accuracy << "%)" << std::endl;

    return 0;
}
