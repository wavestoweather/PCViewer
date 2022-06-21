#pragma once

#include <vulkan/vulkan.h>

struct VkPointer{
    VkBuffer buffer;            // current main buffer bound
    VkDeviceMemory memory;      // memory back for which the pointer is made
    VkDeviceSize byteOffsetBuff;// byte offset from buffer
    VkDeviceSize byteOffsetMem; // byte offset from memory

    VkPointer& operator+=(VkDeviceSize off){
        byteOffsetBuff += off;
        byteOffsetMem += off;
        return *this;
    }
    
    VkPointer operator+(VkDeviceSize off) const{
        return {buffer, memory, byteOffsetBuff + off, byteOffsetMem + off};
    }
};