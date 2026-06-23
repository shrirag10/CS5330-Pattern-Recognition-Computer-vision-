#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "threshold.h"
#include "cleanup.h"
#include "segment.h"
#include "features.h"

namespace fs = std::filesystem;

int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,
                        float minE1, float maxE1, float minE2, float maxE2, int debug);

struct CNNEntry {
    std::string label;
    std::vector<float> embedding;
};

std::vector<CNNEntry> loadCNNDatabase(const std::string &dbPath) {
    std::vector<CNNEntry> db;
    std::ifstream file(dbPath);
    if (!file.is_open()) return db;
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        CNNEntry entry;
        std::getline(ss, entry.label, ',');
        while (std::getline(ss, token, ','))
            entry.embedding.push_back(std::stof(token));
        if (!entry.label.empty() && !entry.embedding.empty())
            db.push_back(entry);
    }
    return db;
}

std::string classifyCNN(const std::vector<float> &query, const std::vector<CNNEntry> &db) {
    double bestSim = -1e9;
    std::string bestLabel = "unknown";

    // compute query norm
    double qNorm = 0.0;
    for (float v : query) qNorm += v * v;
    qNorm = std::sqrt(qNorm);

    for (const auto &entry : db) {
        double dot = 0.0, eNorm = 0.0;
        for (size_t i = 0; i < query.size(); i++) {
            dot += query[i] * entry.embedding[i];
            eNorm += entry.embedding[i] * entry.embedding[i];
        }
        eNorm = std::sqrt(eNorm);
        double sim = (qNorm > 0 && eNorm > 0) ? dot / (qNorm * eNorm) : 0.0;
        if (sim > bestSim) {
            bestSim = sim;
            bestLabel = entry.label;
        }
    }
    return bestLabel;
}

int main(int argc, char *argv[]) {
    std::string datasetDir = "test_data01";
    std::string dbPath = "database.csv";
    std::string modelPath = "resnet18.onnx";
    bool useCNN = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) datasetDir = argv[++i];
        else if (arg == "--db" && i + 1 < argc) dbPath = argv[++i];
        else if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--cnn") useCNN = true;
        else if (arg == "--classical") useCNN = false;
    }

    // load DB and collect labels
    std::vector<TrainingEntry> classicalDB;
    std::vector<CNNEntry> cnnDB;
    std::vector<std::string> labels;

    if (useCNN) {
        cnnDB = loadCNNDatabase(dbPath);
        if (cnnDB.empty()) { std::cerr << "[Error] Could not load CNN DB: " << dbPath << std::endl; return 1; }
        for (const auto &e : cnnDB) {
            bool found = false;
            for (const auto &l : labels) if (l == e.label) { found = true; break; }
            if (!found) labels.push_back(e.label);
        }
        std::cout << "[Info] CNN mode — loaded " << cnnDB.size() << " entries" << std::endl;
    } else {
        classicalDB = loadDatabase(dbPath);
        if (classicalDB.empty()) { std::cerr << "[Error] Could not load DB: " << dbPath << std::endl; return 1; }
        for (const auto &e : classicalDB) {
            bool found = false;
            for (const auto &l : labels) if (l == e.label) { found = true; break; }
            if (!found) labels.push_back(e.label);
        }
        std::cout << "[Info] Classical mode — loaded " << classicalDB.size() << " entries" << std::endl;
    }
    std::sort(labels.begin(), labels.end());
    int N = (int)labels.size();

    auto labelIndex = [&](const std::string &l) {
        for (int i = 0; i < N; i++) if (labels[i] == l) return i;
        return -1;
    };

    cv::dnn::Net net;
    if (useCNN) {
        net = cv::dnn::readNetFromONNX(modelPath);
        std::cout << "[Info] Loaded model: " << modelPath << std::endl;
    }

    std::vector<std::vector<int>> matrix(N, std::vector<int>(N, 0));
    int total = 0, correct = 0, skipped = 0;

    for (const auto &classEntry : fs::directory_iterator(datasetDir)) {
        if (!classEntry.is_directory()) continue;
        std::string trueLabel = classEntry.path().filename().string();
        int trueIdx = labelIndex(trueLabel);
        if (trueIdx < 0) { std::cerr << "[Skip] Label not in DB: " << trueLabel << std::endl; continue; }

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

            std::vector<RegionFeatures> feats = computeRegionFeatures(labelImg, numRegions);
            RegionFeatures largest = feats[0];
            for (const auto &rf : feats)
                if (rf.area > largest.area) largest = rf;

            std::string predicted;
            if (useCNN) {
                cv::Mat embimage, embedding;
                prepEmbeddingImage(img, embimage,
                                   (int)largest.centroid.x, (int)largest.centroid.y,
                                   (float)largest.orientation,
                                   largest.minE1, largest.maxE1,
                                   largest.minE2, largest.maxE2, 0);
                getEmbedding(embimage, embedding, net, 0);
                std::vector<float> embVec(embedding.ptr<float>(0),
                                          embedding.ptr<float>(0) + embedding.total());
                predicted = classifyCNN(embVec, cnnDB);
            } else {
                predicted = classifyFeatures(largest.featureVec, classicalDB);
            }

            int predIdx = labelIndex(predicted);
            if (predIdx >= 0) matrix[trueIdx][predIdx]++;
            if (predicted == trueLabel) correct++;
            total++;

            std::cout << "[" << (predicted == trueLabel ? "OK" : "MISS") << "] "
                      << trueLabel << " -> " << predicted
                      << " (" << imgEntry.path().filename().string() << ")" << std::endl;
        }
    }

    std::cout << "\nConfusion Matrix (rows=true, cols=predicted):\n\n";
    int col_w = 12;
    std::cout << std::string(col_w, ' ');
    for (const auto &l : labels) std::cout << std::setw(col_w) << l;
    std::cout << "\n" << std::string(col_w * (N + 1), '-') << "\n";
    for (int i = 0; i < N; i++) {
        std::cout << std::setw(col_w) << labels[i];
        for (int j = 0; j < N; j++) std::cout << std::setw(col_w) << matrix[i][j];
        std::cout << "\n";
    }

    double accuracy = total > 0 ? (100.0 * correct / total) : 0.0;
    std::cout << "\nAccuracy: " << correct << "/" << total
              << " (" << accuracy << "%)" << std::endl;

    return 0;
}
