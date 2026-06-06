#pragma once
#include "Featurizer.h"
#include <algorithm>  // std::min
#include <cmath>      // std::sqrt
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <string>

// Shared helper: computes normalized 3D BGR histogram for a region
inline std::vector<float> compute3DHistogram(const cv::Mat& image, int bins) {
    int total = image.rows * image.cols;
    if (total == 0) return std::vector<float>(bins * bins * bins, 0.0f); // guard: empty image

    float binWidth = 256.0f / bins;
    std::vector<float> hist(bins * bins * bins, 0.0f);

    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            cv::Vec3b px = image.at<cv::Vec3b>(r, c);
            int bBin = std::min((int)(px[0] / binWidth), bins - 1);
            int gBin = std::min((int)(px[1] / binWidth), bins - 1);
            int rBin = std::min((int)(px[2] / binWidth), bins - 1);
            hist[bBin * bins * bins + gBin * bins + rBin]++;
        }
    }
    for (float& v : hist) v /= total;
    return hist;
}

class BaselineFeaturizer : public Featurizer {
public:
    std::vector<float> featurize(const NamedImage& img) override {
        const cv::Mat& image = img.mat;
        int cx = image.cols / 2;
        int cy = image.rows / 2;

        // guard: image must be at least 7x7 for the 3x3 pixel patch
        if (image.cols < 7 || image.rows < 7) {
            std::cerr << "BaselineFeaturizer: image '" << img.name << "' too small (" 
                      << image.cols << "x" << image.rows << "), returning zeros\n";
            return std::vector<float>(147, 0.0f);
        }

        std::vector<float> features;
        for (int r = cy - 3; r <= cy + 3; r++) {
            for (int c = cx - 3; c <= cx + 3; c++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);
                features.push_back(pixel[0]);
                features.push_back(pixel[1]);
                features.push_back(pixel[2]);
            }
        }
        return features; // 147 values
    }
};

class HistogramFeaturizer : public Featurizer {
public:
    HistogramFeaturizer(int bins = 16) : bins(bins) {}

    std::vector<float> featurize(const NamedImage& img) override {
        return compute3DHistogram(img.mat, bins);
    }

    // exposed for MultiHistogramFeaturizer composition
    std::vector<float> featurizeRegion(const cv::Mat& region) {
        return compute3DHistogram(region, bins);
    }

private:
    int bins;
};

class MultiHistogramFeaturizer : public Featurizer {
public:
    MultiHistogramFeaturizer(int bins = 8) : regionHistogram(bins) {}

    std::vector<float> featurize(const NamedImage& img) override {
        const cv::Mat& image = img.mat;
        int W = image.cols, H = image.rows;
        int rw = W / 3, rh = H / 3;

        std::vector<float> features;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int x = col * rw, y = row * rh;
                int w = (col == 2) ? W - x : rw;
                int h = (row == 2) ? H - y : rh;
                cv::Mat region = image(cv::Rect(x, y, w, h));
                std::vector<float> hist = regionHistogram.featurizeRegion(region);
                features.insert(features.end(), hist.begin(), hist.end());
            }
        }
        return features; // 9 * bins^3 values (default: 9 * 512 = 4608)
    }

private:
    HistogramFeaturizer regionHistogram;
};

// Feature vector = [color_hist (512) | gradient_mag_hist (16)] = 528 values
// Sobel applied manually on grayscale — no cv::Sobel or cv::calcHist used
class TextureColorFeaturizer : public Featurizer {
public:
    TextureColorFeaturizer(int colorBins = 8, int textureBins = 16)
        : colorHistogram(colorBins), textureBins(textureBins) {}

    std::vector<float> featurize(const NamedImage& img) override {
        std::vector<float> colorHist   = colorHistogram.featurizeRegion(img.mat);
        std::vector<float> textureHist = computeGradMagHist(img.mat);
        colorHist.insert(colorHist.end(), textureHist.begin(), textureHist.end());
        return colorHist;
    }

private:
    HistogramFeaturizer colorHistogram;
    int textureBins;

    // Sobel gradient magnitude histogram over grayscale, normalized
    // Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]   Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
    std::vector<float> computeGradMagHist(const cv::Mat& image) {
        int H = image.rows, W = image.cols;
        std::vector<float> gray(H * W);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++) {
                cv::Vec3b px = image.at<cv::Vec3b>(r, c);
                gray[r * W + c] = 0.114f * px[0] + 0.587f * px[1] + 0.299f * px[2];
            }

        // max magnitude: sqrt((2*255)^2 + (4*255)^2) = 510*sqrt(5) ~= 1140.39
        float maxMag = 510.0f * std::sqrt(5.0f);
        std::vector<float> hist(textureBins, 0.0f);
        int count = 0;

        for (int r = 1; r < H - 1; r++) {
            for (int c = 1; c < W - 1; c++) {
                float gx = -gray[(r-1)*W+(c-1)] + gray[(r-1)*W+(c+1)]
                           -2.0f*gray[r*W+(c-1)] + 2.0f*gray[r*W+(c+1)]
                           -gray[(r+1)*W+(c-1)] + gray[(r+1)*W+(c+1)];
                float gy = -gray[(r-1)*W+(c-1)] - 2.0f*gray[(r-1)*W+c] - gray[(r-1)*W+(c+1)]
                           +gray[(r+1)*W+(c-1)] + 2.0f*gray[(r+1)*W+c] + gray[(r+1)*W+(c+1)];
                float mag = std::sqrt(gx * gx + gy * gy);
                int bin = std::min((int)(mag / maxMag * textureBins), textureBins - 1);
                hist[bin]++;
                count++;
            }
        }
        if (count > 0)
            for (float& v : hist) v /= count;
        return hist;
    }
};

// Loads precomputed ResNet18 embeddings from CSV at construction time.
// CSV format: filename,val1,val2,...,val512
class DnnFeaturizer : public Featurizer {
public:
    DnnFeaturizer(const std::string& csvPath) {
        std::ifstream file(csvPath);
        if (!file.is_open()) {
            std::cerr << "DnnFeaturizer: could not open CSV: " << csvPath << "\n";
            return;
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

            std::string fname = tokens[0];
            // trim leading whitespace
            auto start = fname.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue; // skip all-whitespace filenames
            fname = fname.substr(start);
            auto end = fname.find_last_not_of(" \t\r\n");
            if (end != std::string::npos) fname = fname.substr(0, end + 1);

            std::vector<float> vec;
            vec.reserve(tokens.size() - 1);
            bool parse_ok = true;
            for (int i = 1; i < (int)tokens.size(); i++) {
                try { vec.push_back(std::stof(tokens[i])); }
                catch (...) { parse_ok = false; break; } // skip malformed rows
            }
            if (parse_ok)
                embeddings[fname] = std::move(vec);
        }
    }

    std::vector<float> featurize(const NamedImage& img) override {
        auto it = embeddings.find(img.name);
        if (it != embeddings.end()) return it->second;
        std::cerr << "DnnFeaturizer: '" << img.name << "' not found in CSV\n";
        return std::vector<float>(512, 0.0f);
    }

    bool has(const std::string& name) const {
        return embeddings.count(name) > 0;
    }

private:
    std::map<std::string, std::vector<float>> embeddings;
};

// Combined [colorHist (512) | dnnEmbedding (512)] = 1024 values
class CustomFeaturizer : public Featurizer {
public:
    CustomFeaturizer(const std::string& csvPath)
        : dnn(csvPath), color(8) {}

    std::vector<float> featurize(const NamedImage& img) override {
        std::vector<float> colorPart = color.featurizeRegion(img.mat);
        std::vector<float> dnnPart   = dnn.featurize(img);
        colorPart.insert(colorPart.end(), dnnPart.begin(), dnnPart.end());
        return colorPart;
    }

    bool hasEmbedding(const std::string& name) const {
        return dnn.has(name);
    }

private:
    DnnFeaturizer dnn;
    HistogramFeaturizer color;
};
