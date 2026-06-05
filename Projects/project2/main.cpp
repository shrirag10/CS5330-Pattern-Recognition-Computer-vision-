#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
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
    
    int N = 5;
    std::string csvPath = dbDir + "/../ResNet18_olym.csv";
    if (argc >= 5) {
        std::string arg4 = argv[4];
        if (arg4 != "bottom") {
            try {
                N = std::stoi(arg4);
                if (argc >= 6 && std::string(argv[5]) != "bottom") {
                    csvPath = argv[5];
                }
            } catch (...) {
                // If N is not a valid integer, it might be the csvPath
                if (arg4.size() >= 4 && arg4.substr(arg4.size() - 4) == ".csv") {
                    csvPath = arg4;
                }
            }
        }
    }

    // pass 'bottom' as any trailing arg to show N least similar instead of most similar
    bool showBottom = false;
    for (int i = 4; i < argc; i++)
        if (std::string(argv[i]) == "bottom") { showBottom = true; break; }

    std::string targetName = fs::path(targetPath).filename().string();

    // =========================================================
    // Feature and Distance Metric Selection
    // =========================================================
    std::map<std::string, std::vector<float>> dnnFeatures;
    if (method == "dnn" || method == "custom") {
        dnnFeatures = loadDNNFeatures(csvPath);
        if (dnnFeatures.empty()) {
            std::cerr << "No DNN features loaded from: " << csvPath << "\n";
            return 1;
        }
        if (dnnFeatures.find(targetName) == dnnFeatures.end()) {
            std::cerr << "Target image '" << targetName << "' not found in CSV.\n";
            return 1;
        }
    }

    cv::Mat targetImg = cv::imread(targetPath, cv::IMREAD_COLOR);
    if (targetImg.empty()) {
        std::cerr << "Could not load target image: " << targetPath << "\n";
        return 1;
    }

    std::unique_ptr<Featurizer>       featurizer;
    std::unique_ptr<SimilarityScoring> scorer;

    if (method == "baseline") {
        featurizer = std::make_unique<BaselineFeaturizer>();
        scorer     = std::make_unique<SSDScoring>();
    } else if (method == "histogram") {
        featurizer = std::make_unique<HistogramFeaturizer>();        // 3D BGR, 16 bins/ch → 4096 values
        scorer     = std::make_unique<HistogramIntersectionScoring>();
    } else if (method == "multihistogram") {
        featurizer = std::make_unique<MultiHistogramFeaturizer>();   // 3x3 grid, 8 bins/ch → 4608 values
        scorer     = std::make_unique<MultiHistogramScoring>();      // equal weight per region
    } else if (method == "multihistogram-weighted") {
        featurizer = std::make_unique<MultiHistogramFeaturizer>();
        scorer     = std::make_unique<MultiHistogramCenterWeightedScoring>(); // center=50%
    } else if (method == "texture") {
        featurizer = std::make_unique<TextureColorFeaturizer>();     // color (512) + sobel mag (16) = 528
        scorer     = std::make_unique<TextureColorScoring>();        // equal-weight histogram intersection
    } else if (method == "dnn") {
        featurizer = std::make_unique<DnnFeaturizer>(dnnFeatures);
        scorer     = std::make_unique<CosineDistanceScoring>();
    } else if (method == "custom") {
        featurizer = std::make_unique<CustomFeaturizer>(dnnFeatures);
        scorer     = std::make_unique<CustomScoring>(512, 512);
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        std::cerr << "Use: baseline, histogram, multihistogram, multihistogram-weighted, texture, dnn, custom\n";
        return 1;
    }

    std::vector<float> targetFeatures = featurizer->featurize(targetImg, targetName);
    std::vector<ImageMatch> matches;

    // Unified matching loop — featurizer encapsulates all details.
    // DnnFeaturizer ignores the cv::Mat and looks up by filename.
    // ClassicFeaturizers ignore the filename and compute from pixels.
    for (const auto& entry : fs::directory_iterator(dbDir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname == targetName) continue;

        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) continue;

        std::vector<float> feat = featurizer->featurize(img, fname);
        matches.push_back({ fname, scorer->score(targetFeatures, feat) });
    }


    // Sort and output top N
    std::sort(matches.begin(), matches.end(), [](const ImageMatch& a, const ImageMatch& b) {
        return a.score < b.score;
    });

    if (showBottom) std::reverse(matches.begin(), matches.end());

    std::cout << (showBottom ? "Bottom " : "Top ") << N << " matches for: " << targetName
              << " (method: " << method << ")\n";
    for (int i = 0; i < N && i < (int)matches.size(); i++)
        std::cout << i + 1 << ". " << matches[i].name
                  << "  (score: " << matches[i].score << ")\n";

    return 0;
}
