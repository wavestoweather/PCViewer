#pragma once

#include <vulkan/vulkan.h>

struct VkPointer{
    VkBuffer buffer;            // current main buffer bound
    VkDeviceMemory memory;      // memory back for which the pointer is made
    VkDeviceSize byteOffset;    // byte offset from buffer

    VkPointer& operator+=(VkDeviceSize off){
        byteOffset += off;
        return *this;
    }
    
    VkPointer operator+(VkDeviceSize off) const{
        return {buffer, memory, byteOffset + off};
    }
};