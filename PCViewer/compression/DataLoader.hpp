#pragma once

#include <vector>
#include <inttypes.h>
#include "../Attribute.hpp"

// interface for a data loader
class DataLoader{
public:
    virtual void dataAnalysis(size_t& dimensionSize, size_t& dataSize, std::vector<Attribute>& attributes) = 0;
    virtual bool getNext(std::vector<float>& d) = 0;
    virtual void reset();
};