#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "features.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    std::string datasetDir = "train_data";
    std::string outputCSV = "database.csv";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) datasetDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputCSV = argv[++i];
    }

    if (!fs::exists(datasetDir)) {
        std::cerr << "[Error] Dataset directory not found: " << datasetDir << std::endl;
        return 1;
    }

    // write header
    std::ofstream csv(outputCSV);
    if (!csv.is_open()) {
        std::cerr << "[Error] Could not open output file: " << outputCSV << std::endl;
        return 1;
    }
    csv << "label,elongation,h1,h2\n";

    int total = 0, skipped = 0;

    for (const auto &classEntry : fs::directory_iterator(datasetDir)) {
        if (!classEntry.is_directory()) continue;
        std::string label = classEntry.path().filename().string();

        for (const auto &imgEntry : fs::directory_iterator(classEntry.path())) {
            std::string ext = imgEntry.path().extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

            cv::Mat img = cv::imread(imgEntry.path().string());
            if (img.empty()) {
                std::cerr << "[Skip] Could not load: " << imgEntry.path() << std::endl;
                skipped++;
                continue;
            }

            // pipeline
            int thresh = computeDynamicThresholdISODATA(img, 4);
            cv::Mat binary, cleaned, labelImg;
            thresholdBinary(img, binary, thresh, true);
            cleanupBinary(binary, cleaned, 5, 3);
            int numRegions = segmentRegions(cleaned, labelImg, 500);

            if (numRegions == 0) {
                std::cerr << "[Skip] No regions found in: " << imgEntry.path() << std::endl;
                skipped++;
                continue;
            }

            std::vector<RegionFeatures> features = computeRegionFeatures(labelImg, numRegions);

            // pick largest region
            RegionFeatures largest = features[0];
            for (const auto &rf : features)
                if (rf.area > largest.area) largest = rf;

            csv << label;
            for (double f : largest.featureVec) csv << "," << f;
            csv << "\n";

            std::cout << "[OK] " << label << " <- " << imgEntry.path().filename().string()
                      << " | thresh=" << thresh << " regions=" << numRegions
                      << " features=[";
            for (size_t i = 0; i < largest.featureVec.size(); i++) {
                std::cout << largest.featureVec[i];
                if (i + 1 < largest.featureVec.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            total++;
        }
    }

    csv.close();
    std::cout << "\nDone. " << total << " samples written to " << outputCSV;
    if (skipped > 0) std::cout << " (" << skipped << " skipped)";
    std::cout << std::endl;
    return 0;
}
