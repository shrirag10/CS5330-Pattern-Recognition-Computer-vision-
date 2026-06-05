#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <opencv2/opencv.hpp>
#include "FeaturizerImpl.h"
#include "SimilarityScoringImpl.h"

namespace fs = std::filesystem;

// result struct — no features stored after scoring to save memory
struct ImageMatch {
    std::string name;
    float score;
};

// Load ResNet18 CSV: maps filename -> 512-dim embedding
// Expected format per row: filename,val1,val2,...,val512
std::map<std::string, std::vector<float>> loadDNNFeatures(const std::string& csvPath) {
    std::map<std::string, std::vector<float>> features;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Could not open CSV: " << csvPath << "\n";
        return features;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ','))
            tokens.push_back(token);
        if (tokens.size() < 2) continue;

        // trim whitespace from filename
        std::string fname = tokens[0];
        fname.erase(0, fname.find_first_not_of(" \t\r\n"));
        fname.erase(fname.find_last_not_of(" \t\r\n") + 1);

        std::vector<float> vec;
        vec.reserve(tokens.size() - 1);
        for (int i = 1; i < (int)tokens.size(); i++)
            vec.push_back(std::stof(tokens[i]));

        features[fname] = std::move(vec);
    }
    return features;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./match <target_image> <database_dir> <method> [N] [csv_path]\n";
        std::cerr << "Methods: baseline, histogram, multihistogram, multihistogram-weighted,\n";
        std::cerr << "         texture, dnn, custom\n";
        std::cerr << "N defaults to 5. csv_path defaults to <database_dir>/../ResNet18_olym.csv\n";
        return 1;
    }

    std::string targetPath = argv[1];
    std::string dbDir      = argv[2];
    std::string method     = argv[3];
    int N                  = (argc >= 5) ? std::stoi(argv[4]) : 5;
    // default CSV sits one level above the olympus/ folder
    std::string csvPath    = (argc >= 6) ? argv[5] : dbDir + "/../ResNet18_olym.csv";

    std::string targetName = fs::path(targetPath).filename().string();

    // =========================================================
    // DNN and Custom methods — features come from CSV, not image
    // =========================================================
    if (method == "dnn" || method == "custom") {
        auto dnnFeatures = loadDNNFeatures(csvPath);
        if (dnnFeatures.empty()) {
            std::cerr << "No DNN features loaded from: " << csvPath << "\n";
            return 1;
        }
        if (dnnFeatures.find(targetName) == dnnFeatures.end()) {
            std::cerr << "Target image '" << targetName << "' not found in CSV.\n";
            return 1;
        }

        const std::vector<float>& targetDNN = dnnFeatures[targetName];
        std::vector<ImageMatch> matches;

        if (method == "dnn") {
            // Task 5: pure cosine distance on DNN embeddings
            CosineDistanceScoring scorer;
            for (const auto& [fname, feat] : dnnFeatures) {
                if (fname == targetName) continue;
                matches.push_back({ fname, scorer.score(targetDNN, feat) });
            }

        } else {
            // Task 7 (custom): whole-image color histogram + DNN embedding, combined distance
            // Feature vector layout: [colorHist (8^3=512) | dnnEmbedding (512)] = 1024 values
            // Distance = 0.4 * (1 - colorIntersect) + 0.6 * cosine_dnn
            HistogramFeaturizer colorizer(8); // 8 bins/channel -> 512 values
            CustomScoring scorer(512, 512);

            cv::Mat targetImg = cv::imread(targetPath);
            if (targetImg.empty()) {
                std::cerr << "Could not load target image: " << targetPath << "\n";
                return 1;
            }

            // build target combined feature: [colorHist | dnn]
            std::vector<float> targetColor = colorizer.featurize(targetImg);
            std::vector<float> targetCombined = targetColor;
            targetCombined.insert(targetCombined.end(), targetDNN.begin(), targetDNN.end());

            for (const auto& [fname, dnnFeat] : dnnFeatures) {
                if (fname == targetName) continue;

                std::string imgPath = dbDir + "/" + fname;
                cv::Mat img = cv::imread(imgPath);
                if (img.empty()) continue;

                std::vector<float> colorFeat = colorizer.featurize(img);
                std::vector<float> combined  = colorFeat;
                combined.insert(combined.end(), dnnFeat.begin(), dnnFeat.end());

                matches.push_back({ fname, scorer.score(targetCombined, combined) });
            }
        }

        std::sort(matches.begin(), matches.end(), [](const ImageMatch& a, const ImageMatch& b) {
            return a.score < b.score;
        });

        std::cout << "Top " << N << " matches for: " << targetName
                  << " (method: " << method << ")\n";
        for (int i = 0; i < N && i < (int)matches.size(); i++)
            std::cout << i + 1 << ". " << matches[i].name
                      << "  (score: " << matches[i].score << ")\n";

        return 0;
    }

    // =========================================================
    // Classic methods — features computed from the image pixels
    // =========================================================
    cv::Mat targetImg = cv::imread(targetPath);
    if (targetImg.empty()) {
        std::cerr << "Could not load target image: " << targetPath << "\n";
        return 1;
    }

    Featurizer*       featurizer = nullptr;
    SimilarityScoring* scorer    = nullptr;

    if (method == "baseline") {
        featurizer = new BaselineFeaturizer();
        scorer     = new SSDScoring();
    } else if (method == "histogram") {
        featurizer = new HistogramFeaturizer();        // 3D BGR, 16 bins/ch → 4096 values
        scorer     = new HistogramIntersectionScoring();
    } else if (method == "multihistogram") {
        featurizer = new MultiHistogramFeaturizer();   // 3x3 grid, 8 bins/ch → 4608 values
        scorer     = new MultiHistogramScoring();      // equal weight per region
    } else if (method == "multihistogram-weighted") {
        featurizer = new MultiHistogramFeaturizer();
        scorer     = new MultiHistogramCenterWeightedScoring(); // center=50%
    } else if (method == "texture") {
        featurizer = new TextureColorFeaturizer();     // color (512) + sobel mag (16) = 528
        scorer     = new TextureColorScoring();        // equal-weight histogram intersection
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        std::cerr << "Use: baseline, histogram, multihistogram, multihistogram-weighted, texture, dnn, custom\n";
        return 1;
    }

    std::vector<float> targetFeatures = featurizer->featurize(targetImg);
    std::vector<ImageMatch> matches;

    for (const auto& entry : fs::directory_iterator(dbDir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname == targetName) continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        std::vector<float> feat = featurizer->featurize(img);
        matches.push_back({ fname, scorer->score(targetFeatures, feat) });
    }

    std::sort(matches.begin(), matches.end(), [](const ImageMatch& a, const ImageMatch& b) {
        return a.score < b.score;
    });

    std::cout << "Top " << N << " matches for: " << targetName
              << " (method: " << method << ")\n";
    for (int i = 0; i < N && i < (int)matches.size(); i++)
        std::cout << i + 1 << ". " << matches[i].name
                  << "  (score: " << matches[i].score << ")\n";

    delete featurizer;
    delete scorer;
    return 0;
}
