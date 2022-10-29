#pragma once
#include "HuffmanTable.h"
#include "GpuInstance.hpp"
#include "../cpuCompression/util.h"

struct RLHuffDecodeDataCpu{
    size_t symbolCount{};

    std::vector<uint32_t> symbolOffsets{};
    std::vector<uint32_t> zeroCountOffsets{};
    vkCompress::HuffmanDecodeTable symbolTable;
    vkCompress::HuffmanDecodeTable zeroCountTable;
    std::vector<uint32_t> codewordStream;        // holds the compressed bits of the main codewords
    std::vector<uint32_t> zeroStream;            // holds the compressed bits of the 0 run lengths(needs zeroCountTable for decompression)
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
        const uint32_t offsetAlignment = 16;

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

        auto [buffers, offsets, mem] = VkUtil::createMultiBufferBound(gpuContext, {wholeSize}, {VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT}, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        buffer = buffers[0];
        memory = mem;   // offsets is not need, as only a single buffer is created

        //uploading all the data
        //VkUtil::uploadData(gpuContext.device, memory, symbolTableOffset, symbolTableSize, cpuData.symbolTable.m_pStorage);
        //if(symbolStreamSize)
        //    VkUtil::uploadData(gpuContext.device, memory, symbolStreamOffset, symbolStreamSize, cpuData.codewordStream.data());
        //VkUtil::uploadData(gpuContext.device, memory, symbolOffsetsOffset, symbolOffsetsSize, cpuData.symbolOffsets.data());
        //VkUtil::uploadData(gpuContext.device, memory, zeroCountTableOffset, zeroCountTableSize, cpuData.zeroCountTable.m_pStorage);
        //if(zeroCountStreamSize)
        //    VkUtil::uploadData(gpuContext.device, memory, zeroCountStreamOffset, zeroCountStreamSize, cpuData.zeroStream.data());
        //VkUtil::uploadData(gpuContext.device, memory, zeroCountOffsetsOffset, zeroCountOffsetSize, cpuData.zeroCountOffsets.data());

        std::vector<uint8_t> gpuData(wholeSize);
        std::memcpy(gpuData.data() + symbolTableOffset, cpuData.symbolTable.m_pStorage, symbolTableSize);
        std::memcpy(gpuData.data() + symbolStreamOffset, cpuData.codewordStream.data(), symbolStreamSize);
        std::memcpy(gpuData.data() + symbolOffsetsOffset, cpuData.symbolOffsets.data(), symbolOffsetsSize);
        std::memcpy(gpuData.data() + zeroCountTableOffset, cpuData.zeroCountTable.m_pStorage, zeroCountTableSize);
        std::memcpy(gpuData.data() + zeroCountStreamOffset, cpuData.zeroStream.data(), zeroCountStreamSize);
        std::memcpy(gpuData.data() + zeroCountOffsetsOffset, cpuData.zeroCountOffsets.data(), zeroCountOffsetSize);
        VkUtil::uploadDataIndirect(gpuContext, buffer, wholeSize, gpuData.data());
    }
    RLHuffDecodeDataGpu(const RLHuffDecodeDataGpu&) = delete;   // no copy
    RLHuffDecodeDataGpu& operator=(const RLHuffDecodeDataGpu&) = delete; // no copy on assignment
    // move constructor and assignment operator are still defined by default
    RLHuffDecodeDataGpu(RLHuffDecodeDataGpu &&o){
        std::memcpy(this, &o, sizeof(o));
        std::memset(&o, 0, sizeof(o));
    };
    RLHuffDecodeDataGpu& operator=(RLHuffDecodeDataGpu &&o){
        std::memcpy(this, &o, sizeof(o));
        std::memset(&o, 0, sizeof(o));
        return *this;
    };

    ~RLHuffDecodeDataGpu(){
        if(buffer)
            vkDestroyBuffer(gpuContext.device, buffer, nullptr);
        if(memory)
            vkFreeMemory(gpuContext.device, memory, nullptr);
    }
};