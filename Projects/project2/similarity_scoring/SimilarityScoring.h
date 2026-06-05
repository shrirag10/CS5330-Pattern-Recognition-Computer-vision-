#pragma once
#include <vector>

class SimilarityScoring {
public:
    virtual float score(const std::vector<float>& a, const std::vector<float>& b) = 0;
    virtual ~SimilarityScoring() = default;
};
