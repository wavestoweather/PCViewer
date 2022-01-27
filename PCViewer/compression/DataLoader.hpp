#pragma once

#include <vector>
#include <inttypes.h>
#include "../Attribute.hpp"

// interface for a data loader
class DataLoader{
public:
    std::vector<QueryAttribute> queryAttributes{};    //are being filled upon creation of the loader and can be changed to filter out some dimensions

    virtual const float& progress() const = 0;
    virtual void dataAnalysis(size_t& dataSize, std::vector<Attribute>& attributes) = 0;
    virtual bool getNext(std::vector<float>& d) = 0;
    virtual bool getNextNormalized(std::vector<float>& d) = 0;
    virtual void reset() = 0;
};