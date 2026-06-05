#pragma once
#include "SimilarityScoring.h"

class SSDScoring : public SimilarityScoring {
public:
    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        float ssd = 0.0f;
        for (int i = 0; i < (int)a.size(); i++) {
            float diff = a[i] - b[i];
            ssd += diff * diff;
        }
        return ssd;
    }
};

class HistogramIntersectionScoring : public SimilarityScoring {
public:
    // returns negative intersection so that lower score = more similar (consistent with SSD sorting)
    // histogram must be normalized before calling — intersection range is then [-1, 0]
    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        float intersection = 0.0f;
        for (int i = 0; i < (int)a.size(); i++)
            intersection += std::min(a[i], b[i]);
        return -intersection;
    }
};

class MultiHistogramScoring : public SimilarityScoring {
public:
    // equal weighting — each of the 9 regions contributes equally
    MultiHistogramScoring(int bins = 8) : regionSize(bins * bins * bins) {}

    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        float total = 0.0f;
        for (int region = 0; region < 9; region++) {
            int start = region * regionSize;
            float intersection = 0.0f;
            for (int i = 0; i < regionSize; i++)
                intersection += std::min(a[start + i], b[start + i]);
            total += intersection;
        }
        return -(total / 9.0f); // negate so lower = more similar
    }

private:
    int regionSize;
};

class MultiHistogramCenterWeightedScoring : public SimilarityScoring {
public:
    // center-heavy weighting: center=0.50, edges=0.075 each, corners=0.05 each
    MultiHistogramCenterWeightedScoring(int bins = 8) : regionSize(bins * bins * bins) {}

    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        static const float weights[9] = {
            0.05f,   // TL corner
            0.075f,  // TC edge
            0.05f,   // TR corner
            0.075f,  // ML edge
            0.50f,   // center
            0.075f,  // MR edge
            0.05f,   // BL corner
            0.075f,  // BC edge
            0.05f    // BR corner
        };

        float total = 0.0f;
        for (int region = 0; region < 9; region++) {
            int start = region * regionSize;
            float intersection = 0.0f;
            for (int i = 0; i < regionSize; i++)
                intersection += std::min(a[start + i], b[start + i]);
            total += weights[region] * intersection;
        }
        return -total; // negate so lower = more similar
    }

private:
    int regionSize;
};

// Task 4: Texture + Color scoring
// Splits the feature vector into color (first colorHistSize values) and texture (remaining textureBins values).
// Computes normalized histogram intersection on each half independently, then averages with equal weight.
// Negated so that lower score = more similar (consistent with all other scorers).
class TextureColorScoring : public SimilarityScoring {
public:
    // colorHistSize = colorBins^3 (default 8^3=512), textureBins = 16
    TextureColorScoring(int colorHistSize = 512, int textureBins = 16)
        : colorHistSize(colorHistSize), textureBins(textureBins) {}

    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        // color sub-histogram intersection
        float colorIntersect = 0.0f;
        for (int i = 0; i < colorHistSize; i++)
            colorIntersect += std::min(a[i], b[i]);

        // texture sub-histogram intersection
        float textureIntersect = 0.0f;
        for (int i = 0; i < textureBins; i++)
            textureIntersect += std::min(a[colorHistSize + i], b[colorHistSize + i]);

        // equal weighting: 0.5 color + 0.5 texture
        float combined = 0.5f * colorIntersect + 0.5f * textureIntersect;
        return -combined; // negate so lower = more similar
    }

private:
    int colorHistSize;
    int textureBins;
};

// Task 5: Cosine distance for DNN embeddings
// d(v1, v2) = 1 - cos(theta)
// cos(theta) = (v1 . v2) / (|v1| * |v2|)
// Range: [0, 2] in theory; [0, 1] for non-negative activations (ReLU outputs)
// 0 = identical direction, 1 = orthogonal, 2 = opposite
class CosineDistanceScoring : public SimilarityScoring {
public:
    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (int i = 0; i < (int)a.size(); i++) {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0f || normB == 0.0f) return 1.0f; // undefined → max distance
        float cosTheta = dot / (std::sqrt(normA) * std::sqrt(normB));
        return 1.0f - cosTheta; // lower = more similar
    }
};

// Task 7: Custom scoring for combined [colorHist | dnnEmbedding] feature vector
// Feature layout: first colorHistSize values = 3D color histogram (normalized)
//                 next dnnSize values = ResNet18 embedding
// Distance = 0.4 * (1 - colorIntersect) + 0.6 * cosine_dnn
// Both components in [0,1], lower = more similar
class CustomScoring : public SimilarityScoring {
public:
    CustomScoring(int colorHistSize = 512, int dnnSize = 512)
        : colorHistSize(colorHistSize), dnnSize(dnnSize) {}

    float score(const std::vector<float>& a, const std::vector<float>& b) override {
        // --- color part: histogram intersection ---
        float colorIntersect = 0.0f;
        for (int i = 0; i < colorHistSize; i++)
            colorIntersect += std::min(a[i], b[i]);
        float colorDist = 1.0f - colorIntersect; // [0, 1], lower = more similar

        // --- DNN part: cosine distance ---
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (int i = 0; i < dnnSize; i++) {
            int idx = colorHistSize + i;
            dot   += a[idx] * b[idx];
            normA += a[idx] * a[idx];
            normB += b[idx] * b[idx];
        }
        float cosineDist = 1.0f;
        if (normA > 0.0f && normB > 0.0f)
            cosineDist = 1.0f - dot / (std::sqrt(normA) * std::sqrt(normB));

        // equal-ish weighting: 40% color appearance, 60% DNN semantics
        return 0.4f * colorDist + 0.6f * cosineDist;
    }

private:
    int colorHistSize;
    int dnnSize;
};
