#pragma once
#include "../robin_hood_map/robin_hood.h"

struct GPUBinCounts{
    struct CountElement{
        VkBuffer buffer;        // vulkan buffer storing the count information
        uint32_t countSize;     // amount of elements in "buffer"
        std::vector<uint32_t> axesSizes;    // sizes of the axes, Mul(axesSizes) = countSize
    };

    robin_hood::unordered_map<std::vector<uint32_t>, CountElement> counts;
};