#pragma once
#include <cstddef>
#include <vulkan/vulkan.h>

#include "../cpuCompression/global.h"

namespace vkCompress {
class GpuInstance;

//TODO make separate versions for encoder and decoder?
struct HuffmanGPUStreamInfo
{
    // raw data
    VkDeviceAddress dpSymbolStream{};

    // encoded data
    VkDeviceAddress dpCodewordStream{};
    VkDeviceAddress dpOffsets{};

    // encoder-only info
    VkDeviceAddress dpEncodeCodewords{};
    VkDeviceAddress dpEncodeCodewordLengths{};

    // decoder-only info
    VkDeviceAddress dpDecodeTable{};
    uint decodeSymbolTableSize{};
    
    // common info
    uint symbolCount{};
};

size_t huffmanGetRequiredMemory(const GpuInstance* pInstance);
bool huffmanInit(GpuInstance* pInstance);
bool huffmanShutdown(GpuInstance* pInstance);

// note: codingBlockSize has to be a power of 2 between 32 and 256
// note: huffmanEncode assumes that dpCodewordStream is already cleared to all zeros!
// note: huffmanDecode does *not* sync on the upload  of pStreamInfos to complete!!
// note: huffmanDecode requires that the length of each symbol stream (dpSymbolStream) is a multiple of codingBlockSize*WARP_SIZE
//       (transpose kernel requires this)
bool huffmanEncode(GpuInstance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize, uint* pCompressedSizeBits);
bool huffmanDecode(GpuInstance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize);
// note: does the same as huffmanDecode, however expects all the data to be available in vulkan buffers and writes the commands into the commands buffer
// note: execution is only done after the commands command buffer is submitted to a queue
bool huffmanDecode(GpuInstance* pInstance, VkCommandBuffer commands, VkDescriptorSet pStreamInfos, uint streamCount, uint codingBlockSize);

}