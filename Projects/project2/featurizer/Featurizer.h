#pragma once
#include <vector>
#include "../models/NamedImage.h"

class Featurizer {
public:
    virtual std::vector<float> featurize(const NamedImage& img) = 0;
    virtual ~Featurizer() = default;
};
