#pragma once
#include "Featurizer.h"
#include <cmath>

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

    std::vector<float> computeRegionHistogram(const cv::Mat& region) {
        float binWidth = 256.0f / bins;
        int total = region.rows * region.cols;
        std::vector<float> hist(bins * bins * bins, 0.0f);

        for (int r = 0; r < region.rows; r++) {
            for (int c = 0; c < region.cols; c++) {
                cv::Vec3b pixel = region.at<cv::Vec3b>(r, c);
                int bBin = std::min((int)(pixel[0] / binWidth), bins - 1);
                int gBin = std::min((int)(pixel[1] / binWidth), bins - 1);
                int rBin = std::min((int)(pixel[2] / binWidth), bins - 1);
                hist[bBin * bins * bins + gBin * bins + rBin]++;
            }
        }
        for (float& v : hist) v /= total;
        return hist;
    }

private:
    int bins;
};

class MultiHistogramFeaturizer : public Featurizer {
public:
    MultiHistogramFeaturizer(int bins = 8) : bins(bins) {}

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
                std::vector<float> hist = computeRegionHistogram(region);
                features.insert(features.end(), hist.begin(), hist.end());
            }
        }

        return features; // 9 * bins^3 values (default: 9 * 512 = 4608)
    }

private:
    std::vector<float> computeRegionHistogram(const cv::Mat& region) {
        float binWidth = 256.0f / bins;
        int total = region.rows * region.cols;
        std::vector<float> hist(bins * bins * bins, 0.0f);

        for (int r = 0; r < region.rows; r++) {
            for (int c = 0; c < region.cols; c++) {
                cv::Vec3b pixel = region.at<cv::Vec3b>(r, c);
                int bBin = std::min((int)(pixel[0] / binWidth), bins - 1);
                int gBin = std::min((int)(pixel[1] / binWidth), bins - 1);
                int rBin = std::min((int)(pixel[2] / binWidth), bins - 1);
                hist[bBin * bins * bins + gBin * bins + rBin]++;
            }
        }
        for (float& v : hist) v /= total;
        return hist;
    }

    int bins;
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
        // max possible magnitude for 8-bit image: sqrt((4*255)^2+(4*255)^2) ~= 1442
        float maxMag = std::sqrt(2.0f) * 4.0f * 255.0f; // ~1441.6

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
        for (float& v : hist) v /= count;
        return hist;
    }
};
