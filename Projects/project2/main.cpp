#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include "featurizer/FeaturizerImpl.h"
#include "similarity_scoring/SimilarityScoringImpl.h"
#include "models/NamedImage.h"
#include "models/ImageMatch.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ./match <target_image> <database_dir> <method> [N] [--embeds <csv>] [bottom]\n";
        std::cerr << "Methods: baseline, histogram, multihistogram, multihistogram-weighted,\n";
        std::cerr << "         texture, dnn, custom\n";
        return 1;
    }

    std::string targetPath = argv[1];
    std::string dbDir      = argv[2];
    std::string method     = argv[3];

    int N = 5;
    bool showBottom = false;
    std::string csvPath;

    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "bottom") {
            showBottom = true;
        } else if (arg == "--embeds" && i + 1 < argc) {
            csvPath = argv[++i];
        } else {
            try { N = std::stoi(arg); } catch (...) {}
        }
    }

    if ((method == "dnn" || method == "custom") && csvPath.empty()) {
        std::cerr << "Error: --embeds <csv> is required for method '" << method << "'\n";
        return 1;
    }

    std::string targetName = fs::path(targetPath).filename().string();

    cv::Mat targetMat = cv::imread(targetPath, cv::IMREAD_COLOR);
    if (targetMat.empty()) {
        std::cerr << "Could not load target image: " << targetPath << "\n";
        return 1;
    }
    NamedImage target{ targetName, targetMat };

    std::unique_ptr<Featurizer>        featurizer;
    std::unique_ptr<SimilarityScoring> scorer;

    if (method == "baseline") {
        featurizer = std::make_unique<BaselineFeaturizer>();
        scorer     = std::make_unique<SSDScoring>();
    } else if (method == "histogram") {
        featurizer = std::make_unique<HistogramFeaturizer>();
        scorer     = std::make_unique<HistogramIntersectionScoring>();
    } else if (method == "multihistogram") {
        featurizer = std::make_unique<MultiHistogramFeaturizer>();
        scorer     = std::make_unique<MultiHistogramScoring>();
    } else if (method == "multihistogram-weighted") {
        featurizer = std::make_unique<MultiHistogramFeaturizer>();
        scorer     = std::make_unique<MultiHistogramCenterWeightedScoring>();
    } else if (method == "texture") {
        featurizer = std::make_unique<TextureColorFeaturizer>();
        scorer     = std::make_unique<TextureColorScoring>();
    } else if (method == "dnn") {
        auto dnn = std::make_unique<DnnFeaturizer>(csvPath);
        if (!dnn->has(targetName)) {
            std::cerr << "Error: target '" << targetName << "' not found in CSV: " << csvPath << "\n";
            return 1;
        }
        featurizer = std::move(dnn);
        scorer     = std::make_unique<CosineDistanceScoring>();
    } else if (method == "custom") {
        auto custom = std::make_unique<CustomFeaturizer>(csvPath);
        if (!custom->hasEmbedding(targetName)) {
            std::cerr << "Error: target '" << targetName << "' not found in CSV: " << csvPath << "\n";
            return 1;
        }
        featurizer = std::move(custom);
        scorer     = std::make_unique<CustomScoring>(512, 512);
    } else {
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }

    std::vector<float> targetFeatures = featurizer->featurize(target);

    std::vector<ImageMatch> matches;
    for (const auto& entry : fs::directory_iterator(dbDir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname == targetName) continue;

        cv::Mat mat = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (mat.empty()) continue;

        NamedImage img{ fname, mat };
        matches.push_back({ fname, scorer->score(targetFeatures, featurizer->featurize(img)) });
    }

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
