#pragma once
#include "HuffmanTable.h"
#include "GpuInstance.hpp"
#include "../cpuCompression/util.h"

struct RLHuffDecodeDataCpu{
    size_t symbolCount{};

    std::vector<uint> symbolOffsets{};
    std::vector<uint> zeroCountOffsets{};
    vkCompress::HuffmanDecodeTable symbolTable;
    vkCompress::HuffmanDecodeTable zeroCountTable;
    std::vector<uint8_t> codewordStream;        // holds the compressed bits of the main codewords
    std::vector<uint8_t> zeroStream;            // holds the compressed bits of the 0 run lengths(needs zeroCountTable for decompression)
};

struct RLHuffDecodeDataGpu{
    // vulkan context from which the decode table has been created (for automatic destruction)
    VkUtil::Context gpuContext{};
    // there is only a single gpu buffer for all decoding information
    // the _xxxOffset variables give the byte offset of the 
    VkBuffer buffer{};
    VkDeviceMemory memory{};

    size_t symbolTableOffset{};
    size_t symbolStreamOffset{};
    size_t symbolOffsetsOffset{};
    size_t zeroCountTableOffset{};
    size_t zeroCountStreamOffset{};
    size_t zeroCountOffsetsOffset{};

    // creates the gpu stuff from cpu stuff
    RLHuffDecodeDataGpu(vkCompress::GpuInstance* pInstance, const RLHuffDecodeDataCpu& cpuData):
        gpuContext(pInstance->vkContext)
    {
        const uint offsetAlignment = 16;

        size_t wholeSize{};
        symbolTableOffset = wholeSize;
        size_t symbolTableSize = cpuData.symbolTable.computeGPUSize(pInstance);
        wholeSize += symbolTableSize;
        wholeSize = cudaCompress::getAlignedSize(wholeSize, offsetAlignment);    // 16 byte alignment for aligned vec4 reads
        
        symbolStreamOffset = wholeSize;
        size_t symbolStreamSize = cpuData.codewordStream.size() * sizeof(cpuData.codewordStream[0]);
        wholeSize += symbolStreamSize;
        wholeSize = cudaCompress::getAlignedSize(wholeSize, offsetAlignment);

        symbolOffsetsOffset = wholeSize;
        size_t symbolOffsetsSize = cpuData.symbolOffsets.size() * sizeof(cpuData.symbolOffsets[0]);
        wholeSize += symbolOffsetsSize;
        wholeSize = cudaCompress::getAlignedSize(wholeSize, offsetAlignment);

        zeroCountTableOffset = wholeSize;
        size_t zeroCountTableSize = cpuData.zeroCountTable.computeGPUSize(pInstance);
        wholeSize += zeroCountTableSize;
        wholeSize = cudaCompress::getAlignedSize(wholeSize, offsetAlignment);
        
        zeroCountStreamOffset = wholeSize;
        size_t zeroCountStreamSize = cpuData.zeroStream.size() * sizeof(cpuData.zeroStream[0]);
        wholeSize += zeroCountStreamSize;
        wholeSize = cudaCompress::getAlignedSize(wholeSize, offsetAlignment);

        zeroCountOffsetsOffset = wholeSize;
        size_t zeroCountOffsetSize = cpuData.zeroCountOffsets.size() * sizeof(cpuData.zeroCountOffsets[0]);
        wholeSize += zeroCountOffsetSize;
        wholeSize = cudaCompress::getAlignedSize(wholeSize, offsetAlignment);

        auto [buffers, offsets, mem] = VkUtil::createMultiBufferBound(gpuContext, {wholeSize}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT}, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        buffer = buffers[0];
        memory = mem;   // offsets is not need, as only a single buffer is created

        //uploading all the data
        VkUtil::uploadData(gpuContext.device, memory, symbolTableOffset, symbolTableSize, cpuData.symbolTable.m_pStorage);
        VkUtil::uploadData(gpuContext.device, memory, symbolStreamOffset, symbolStreamSize, cpuData.codewordStream.data());
        VkUtil::uploadData(gpuContext.device, memory, symbolOffsetsOffset, symbolOffsetsSize, cpuData.symbolOffsets.data());
        VkUtil::uploadData(gpuContext.device, memory, zeroCountTableOffset, zeroCountTableSize, cpuData.zeroCountTable.m_pStorage);
        VkUtil::uploadData(gpuContext.device, memory, zeroCountStreamOffset, zeroCountStreamSize, cpuData.zeroStream.data());
        VkUtil::uploadData(gpuContext.device, memory, zeroCountOffsetsOffset, zeroCountOffsetSize, cpuData.zeroCountOffsets.data());
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