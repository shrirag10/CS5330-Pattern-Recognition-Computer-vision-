#pragma once
#include "Featurizer.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

class BaselineFeaturizer : public Featurizer {
public:
    std::vector<float> featurize(const cv::Mat& image) override {
        int cx = image.cols / 2;
        int cy = image.rows / 2;

        std::vector<float> features;
        for (int r = cy - 3; r <= cy + 3; r++) {
            for (int c = cx - 3; c <= cx + 3; c++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);
                features.push_back(pixel[0]); // B
                features.push_back(pixel[1]); // G
                features.push_back(pixel[2]); // R
            }
        }
        return features; // 147 values
    }
};

class HistogramFeaturizer : public Featurizer {
public:
    HistogramFeaturizer(int bins = 16) : bins(bins) {}

    std::vector<float> featurize(const cv::Mat& image) override {
        float binWidth = 256.0f / bins;
        int total = image.rows * image.cols;

        // 3D histogram: hist[b][g][r] = count of pixels in that bin combination
        std::vector<float> hist(bins * bins * bins, 0.0f);

        for (int r = 0; r < image.rows; r++) {
            for (int c = 0; c < image.cols; c++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);
                int bBin = std::min((int)(pixel[0] / binWidth), bins - 1);
                int gBin = std::min((int)(pixel[1] / binWidth), bins - 1);
                int rBin = std::min((int)(pixel[2] / binWidth), bins - 1);
                hist[bBin * bins * bins + gBin * bins + rBin]++;
            }
        }

        // normalize by total pixels
        for (float& v : hist) v /= total;

        return hist; // bins^3 values (default: 512)
    }

    // exposed so MultiHistogramFeaturizer can reuse it directly
    std::vector<float> computeRegionHistogram(const cv::Mat& region) {
        return featurize(region);
    }

private:
    int bins;
};

class MultiHistogramFeaturizer : public Featurizer {
public:
    // uses HistogramFeaturizer for region histograms — no code duplication
    MultiHistogramFeaturizer(int bins = 8) : regionHistogram(bins) {}

    std::vector<float> featurize(const cv::Mat& image) override {
        int W = image.cols;
        int H = image.rows;
        int rw = W / 3;  // region width
        int rh = H / 3;  // region height

        std::vector<float> features;

        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                int x = col * rw;
                int y = row * rh;
                // last row/col absorbs remainder pixels
                int w = (col == 2) ? W - x : rw;
                int h = (row == 2) ? H - y : rh;

                cv::Mat region = image(cv::Rect(x, y, w, h));
                std::vector<float> hist = regionHistogram.computeRegionHistogram(region);
                features.insert(features.end(), hist.begin(), hist.end());
            }
        }

        return features; // 9 * bins^3 values (default: 9 * 512 = 4608)
    }

private:
    HistogramFeaturizer regionHistogram;
};

// Task 4: Texture + Color featurizer
// Feature vector = [color_hist (512) | gradient_mag_hist (16)] = 528 values
// Color histogram: 3D RGB, 8 bins/channel, normalized by pixel count
// Texture histogram: Sobel gradient magnitude, 16 bins, normalized by pixel count
// Sobel is applied manually on grayscale — no cv::Sobel or cv::calcHist used
class TextureColorFeaturizer : public Featurizer {
public:
    TextureColorFeaturizer(int colorBins = 8, int textureBins = 16)
        : colorBins(colorBins), textureBins(textureBins) {}

    std::vector<float> featurize(const cv::Mat& image) override {
        std::vector<float> colorHist  = computeColorHist(image);
        std::vector<float> textureHist = computeGradMagHist(image);

        // concatenate: [color | texture]
        std::vector<float> features;
        features.insert(features.end(), colorHist.begin(),   colorHist.end());
        features.insert(features.end(), textureHist.begin(), textureHist.end());
        return features;
    }

private:
    int colorBins;
    int textureBins;

    // whole-image 3D BGR color histogram, normalized
    std::vector<float> computeColorHist(const cv::Mat& image) {
        float binWidth = 256.0f / colorBins;
        int total = image.rows * image.cols;
        std::vector<float> hist(colorBins * colorBins * colorBins, 0.0f);

        for (int r = 0; r < image.rows; r++) {
            for (int c = 0; c < image.cols; c++) {
                cv::Vec3b px = image.at<cv::Vec3b>(r, c);
                int bBin = std::min((int)(px[0] / binWidth), colorBins - 1);
                int gBin = std::min((int)(px[1] / binWidth), colorBins - 1);
                int rBin = std::min((int)(px[2] / binWidth), colorBins - 1);
                hist[bBin * colorBins * colorBins + gBin * colorBins + rBin]++;
            }
        }
        for (float& v : hist) v /= total;
        return hist;
    }

    // Sobel gradient magnitude histogram over grayscale image, normalized
    // Sobel kernels applied manually — no cv::Sobel or cv::calcHist
    std::vector<float> computeGradMagHist(const cv::Mat& image) {
        // convert to grayscale manually: Y = 0.114*B + 0.587*G + 0.299*R
        int H = image.rows;
        int W = image.cols;
        std::vector<float> gray(H * W);
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < W; c++) {
                cv::Vec3b px = image.at<cv::Vec3b>(r, c);
                gray[r * W + c] = 0.114f * px[0] + 0.587f * px[1] + 0.299f * px[2];
            }
        }

        // Sobel 3x3 kernels:
        // Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]   Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
        // skip 1-pixel border
        // Max actual possible magnitude: sqrt((2*255)^2 + (4*255)^2) = 510 * sqrt(5) ~= 1140.39f
        // (occurs e.g. when Gx = 1020, Gy = 510)
        float maxMag = 510.0f * std::sqrt(5.0f); // ~1140.39f

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
        if (count > 0) {
            for (float& v : hist) v /= count;
        }
        return hist;
    }
};

// Task 5: DNN Embedding Featurizer (loads precomputed embeddings from CSV features map)
class DnnFeaturizer : public Featurizer {
public:
    DnnFeaturizer(const std::map<std::string, std::vector<float>>& dnnFeatures)
        : dnnFeatures(dnnFeatures) {}

    // Should always be called with a filename — warn if not
    std::vector<float> featurize(const cv::Mat& /*image*/) override {
        std::cerr << "DnnFeaturizer: featurize() called without filename — returning zero vector\n";
        return std::vector<float>(512, 0.0f);
    }

    std::vector<float> featurize(const cv::Mat& /*image*/, const std::string& filename) override {
        auto it = dnnFeatures.find(filename);
        if (it != dnnFeatures.end()) {
            return it->second;
        }
        // filename not in CSV — return zeros (image will score poorly, not crash)
        std::cerr << "DnnFeaturizer: '" << filename << "' not found in CSV — returning zero vector\n";
        return std::vector<float>(512, 0.0f);
    }

private:
    const std::map<std::string, std::vector<float>>& dnnFeatures;
};

// Task 7: Custom Combined color + DNN Featurizer
class CustomFeaturizer : public Featurizer {
public:
    CustomFeaturizer(const std::map<std::string, std::vector<float>>& dnnFeatures)
        : dnnFeaturizer(dnnFeatures), colorFeaturizer(8) {}

    std::vector<float> featurize(const cv::Mat& image) override {
        return featurize(image, "");
    }

    std::vector<float> featurize(const cv::Mat& image, const std::string& filename) override {
        std::vector<float> colorPart = colorFeaturizer.featurize(image);
        std::vector<float> dnnPart   = dnnFeaturizer.featurize(image, filename);
        std::vector<float> combined  = colorPart;
        combined.insert(combined.end(), dnnPart.begin(), dnnPart.end());
        return combined;
    }

private:
    DnnFeaturizer dnnFeaturizer;
    HistogramFeaturizer colorFeaturizer;
};
