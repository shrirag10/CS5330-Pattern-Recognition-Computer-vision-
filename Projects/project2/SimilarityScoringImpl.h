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
