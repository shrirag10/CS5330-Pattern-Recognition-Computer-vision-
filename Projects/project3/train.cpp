#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "features.h"

namespace fs = std::filesystem;

// declared in utilities.cpp
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,
                        float minE1, float maxE1, float minE2, float maxE2, int debug);

int main(int argc, char *argv[]) {
    std::string datasetDir = "train_data";
    std::string outputCSV = "database.csv";
    std::string modelPath = "resnet18.onnx";
    bool useCNN = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) datasetDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc) outputCSV = argv[++i];
        else if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--cnn") useCNN = true;
        else if (arg == "--classical") useCNN = false;
    }

    if (!fs::exists(datasetDir)) {
        std::cerr << "[Error] Dataset directory not found: " << datasetDir << std::endl;
        return 1;
    }

    cv::dnn::Net net;
    if (useCNN) {
        if (!fs::exists(modelPath)) {
            std::cerr << "[Error] ONNX model not found: " << modelPath << std::endl;
            return 1;
        }
        net = cv::dnn::readNetFromONNX(modelPath);
        std::cout << "[Info] CNN mode — loaded model: " << modelPath << std::endl;
    } else {
        std::cout << "[Info] Classical mode — using moment-based features" << std::endl;
    }

    std::ofstream csv(outputCSV);
    if (!csv.is_open()) {
        std::cerr << "[Error] Could not open output file: " << outputCSV << std::endl;
        return 1;
    }

    if (useCNN)
        csv << "label,embedding\n"; // embedding is a flat vector written as comma-separated values
    else
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

            // pipeline: threshold -> cleanup -> segment -> features
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

            std::vector<RegionFeatures> feats = computeRegionFeatures(labelImg, numRegions);
            RegionFeatures largest = feats[0];
            for (const auto &rf : feats)
                if (rf.area > largest.area) largest = rf;

            csv << label;

            if (useCNN) {
                cv::Mat embimage, embedding;
                prepEmbeddingImage(img, embimage,
                                   (int)largest.centroid.x, (int)largest.centroid.y,
                                   (float)largest.orientation,
                                   largest.minE1, largest.maxE1,
                                   largest.minE2, largest.maxE2, 0);
                getEmbedding(embimage, embedding, net, 0);

                // flatten embedding and write to csv
                float *data = embedding.ptr<float>(0);
                int dims = embedding.total();
                for (int d = 0; d < dims; d++) csv << "," << data[d];
                std::cout << "[OK] " << label << " <- " << imgEntry.path().filename().string()
                          << " | embedding dims=" << dims << std::endl;
            } else {
                for (double f : largest.featureVec) csv << "," << f;
                std::cout << "[OK] " << label << " <- " << imgEntry.path().filename().string()
                          << " | thresh=" << thresh << " regions=" << numRegions
                          << " features=[";
                for (size_t i = 0; i < largest.featureVec.size(); i++) {
                    std::cout << largest.featureVec[i];
                    if (i + 1 < largest.featureVec.size()) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            csv << "\n";
            total++;
        }
    }

    csv.close();
    std::cout << "\nDone. " << total << " samples written to " << outputCSV;
    if (skipped > 0) std::cout << " (" << skipped << " skipped)";
    std::cout << std::endl;
    return 0;
}
