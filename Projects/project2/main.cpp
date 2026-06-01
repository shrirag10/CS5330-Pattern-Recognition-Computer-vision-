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
    if (argc < 3) {
        std::cerr << "Usage: ./featurize <target_image> <database_dir>\n";
        return 1;
    }

    std::string targetPath = argv[1];
    std::string dbDir = argv[2];

    cv::Mat targetImg = cv::imread(targetPath);
    if (targetImg.empty()) {
        std::cerr << "Could not load target image: " << targetPath << "\n";
        return 1;
    }

    BaselineFeaturizer featurizer;
    SSDScoring scorer;

    std::vector<float> targetFeatures = featurizer.featurize(targetImg);

    std::vector<ImageMatch> matches;

    for (const auto& entry : fs::directory_iterator(dbDir)) {
        if (!entry.is_regular_file()) continue;

        std::string path = entry.path().string();
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        ImageMatch match;
        match.name = entry.path().filename().string();
        match.features = featurizer.featurize(img);
        match.score = scorer.score(targetFeatures, match.features);
        matches.push_back(match);
    }

    std::sort(matches.begin(), matches.end(), [](const ImageMatch& a, const ImageMatch& b) {
        return a.score < b.score;
    });

    std::cout << "Top 5 matches for: " << fs::path(targetPath).filename().string() << "\n";
    for (int i = 0; i < 5 && i < (int)matches.size(); i++) {
        std::cout << i + 1 << ". " << matches[i].name << "  (SSD: " << matches[i].score << ")\n";
    }

    return 0;
}
