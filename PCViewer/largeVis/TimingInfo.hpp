#pragma once
#include <vulkan/vulkan.h>

struct TimingInfo{
    VkQueryPool queryPool;
    uint32_t startIndex, endIndex;
};