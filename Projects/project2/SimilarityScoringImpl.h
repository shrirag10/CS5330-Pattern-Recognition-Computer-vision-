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
