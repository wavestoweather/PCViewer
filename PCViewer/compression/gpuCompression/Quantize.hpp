#pragma once

#include "GpuInstance.hpp"

namespace vkCompress{

enum EQuantizeType
{
    QUANTIZE_DEADZONE = 0, // midtread quantizer with twice larger zero bin
    QUANTIZE_UNIFORM,      // standard uniform midtread quantizer
    QUANTIZE_COUNT
};

void unquantizeFromSymbols(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstFloatBufferAddress, VkDeviceAddress srcShortBufferAddress, uint size, float quantizationStep, EQuantizeType quantType = QUANTIZE_DEADZONE);

}
