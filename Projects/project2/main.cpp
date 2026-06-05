#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "FeaturizerImpl.h"
#include "SimilarityScoringImpl.h"

namespace fs = std::filesystem;

struct ImageMatch {
    std::string name;
    std::vector<float> features;
    float score;
};

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./match <target_image> <database_dir> <method>\n";
        std::cerr << "Methods: baseline, histogram, multihistogram, multihistogram-weighted, texture\n";
        return 1;
    }

    std::string targetPath = argv[1];
    std::string dbDir = argv[2];
    std::string method = argv[3];

    cv::Mat targetImg = cv::imread(targetPath);
    if (targetImg.empty()) {
        std::cerr << "Could not load target image: " << targetPath << "\n";
        return 1;
    }

    // pick featurizer and scorer based on method
    Featurizer* featurizer = nullptr;
    SimilarityScoring* scorer = nullptr;

    if (method == "baseline") {
        featurizer = new BaselineFeaturizer();
        scorer = new SSDScoring();
    } else if (method == "histogram") {
        featurizer = new HistogramFeaturizer();
        scorer = new HistogramIntersectionScoring();
    } else if (method == "multihistogram") {
        featurizer = new MultiHistogramFeaturizer();
        scorer = new MultiHistogramScoring();
    } else if (method == "multihistogram-weighted") {
        featurizer = new MultiHistogramFeaturizer();
        scorer = new MultiHistogramCenterWeightedScoring();
    } else if (method == "texture") {
        featurizer = new TextureColorFeaturizer();   // color (512) + sobel mag (16) = 528 values
        scorer = new TextureColorScoring();          // equal-weight histogram intersection
    } else {
        std::cerr << "Unknown method: " << method << ". Use 'baseline', 'histogram', 'multihistogram', 'multihistogram-weighted', or 'texture'.\n";
        return 1;
    }

    std::vector<float> targetFeatures = featurizer->featurize(targetImg);
    std::string targetName = fs::path(targetPath).filename().string();

    std::vector<ImageMatch> matches;

    for (const auto& entry : fs::directory_iterator(dbDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().filename().string() == targetName) continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        ImageMatch match;
        match.name = entry.path().filename().string();
        match.features = featurizer->featurize(img);
        match.score = scorer->score(targetFeatures, match.features);
        matches.push_back(match);
    }

    std::sort(matches.begin(), matches.end(), [](const ImageMatch& a, const ImageMatch& b) {
        return a.score < b.score;
    });

    std::cout << "Top 5 matches for: " << targetName << " (method: " << method << ")\n";
    for (int i = 0; i < 5 && i < (int)matches.size(); i++) {
        std::cout << i + 1 << ". " << matches[i].name << "  (score: " << matches[i].score << ")\n";
    }

    delete featurizer;
    delete scorer;

    return 0;
}
