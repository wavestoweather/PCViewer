#pragma once

#include "GpuInstance.hpp"

namespace vkCompress{
    void dwtFloatInverse(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstAddress, VkDeviceAddress srcAddress, uint size, uint dstRowPitch = 0, uint srcRowPitch = 0);
    void dwtFloatToHalfInverse(GpuInstance* pInstance, VkCommandBuffer commands, VkDeviceAddress dstAddress, VkDeviceAddress srcAddress, uint size, uint dstRowPitch = 0, uint srcRowPitch = 0);
}