#pragma once
#include "HuffmanTable.h"

struct RLHuffDecodeDataCpu{
    size_t symbolCount{};

    std::vector<uint> symbolOffsets{};
    std::vector<uint> zeroCountOffsets{};
    vkCompress::HuffmanDecodeTable decodeTable;
    vkCompress::HuffmanDecodeTable symbolTable;
    vkCompress::HuffmanDecodeTable zeroCountTable;
    std::vector<uint8_t> codewordStream;        // holds the compressed bits of the main codewords
    std::vector<uint8_t> zeroStream;            // holds the compressed bits of the 0 run lengths(needs zeroCountTable for decompression)
};

struct RLHuffDecodeDataGpu{
    // vulkan context from which the decode table has been created (for automatic destruction)
    VkUtil::Context gpuContext;
    // there is only a single gpu buffer for all decoding information
    // the _xxxOffset variables give the byte offset of the 
    VkBuffer buffer{};
    VkDeviceMemory memory{};

    size_t decodeTableOffset{};
    size_t codewordStreamOffset{};
    size_t symbolTableOffset{};
    size_t symbolOffsetsOffset{};
    size_t zeroCountTableOffset{};
    size_t zeroCountOffsetsOffset{};

    // creates the gpu stuff from cpu stuff
    RLHuffDecodeDataGpu(const RLHuffDecodeDataCpu& cpuData){
        // TODO: 
    }
    RLHuffDecodeDataGpu(const RLHuffDecodeDataGpu&) = delete;   // no copy
    RLHuffDecodeDataGpu& operator=(const RLHuffDecodeDataGpu&) = delete; // no copy on assignment
    // move constructor and assignment operator are still defined by default

    ~RLHuffDecodeDataGpu(){
        if(buffer)
            vkDestroyBuffer(gpuContext.device, buffer, nullptr);
        if(memory)
            vkFreeMemory(gpuContext.device, memory, nullptr);
    }
};