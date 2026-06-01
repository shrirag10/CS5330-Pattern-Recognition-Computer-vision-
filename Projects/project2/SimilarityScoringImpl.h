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
