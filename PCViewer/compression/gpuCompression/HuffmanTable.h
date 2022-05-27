#pragma once

#include "../../VkUtil.h"
#include "../cpuCompression/EncodeCommon.h"
#include "../cpuCompression/BitStream.h"
#include "../cpuCompression/global.h"
#include "GpuInstance.hpp"

namespace vkCompress{

class HuffmanDecodeTable
{
public:
    HuffmanDecodeTable(const GpuInstance* pInstance);
    HuffmanDecodeTable(HuffmanDecodeTable&& other);
    ~HuffmanDecodeTable();

    HuffmanDecodeTable& operator=(HuffmanDecodeTable&& other);

    void clear();
    void readFromBitStream(const GpuInstance* pInstance, BitStreamReadOnly& bitstream);

    uint getSymbolTableSize() const { return m_symbolTableSize; }

    static uint computeMaxGPUSize(const GpuInstance* pInstance);
    uint computeGPUSize(const VGpuInstance* pInstance) const;
    void copyToBuffer(const GpuInstance* pInstance, byte* pTable) const;
    void uploadToGPU(const GpuInstance* pInstance, byte* dpTable) const;
    void uploadToGPUAsync(const VGpuInstancet* pInstance, byte* dpTable) const;
    void syncOnLastAsyncUpload() const;

private:
    cudaCompress::byte* m_pStorage;

    // indexed by codeword length
    // these are just pointers into m_pCodewordIndex
    int* m_pCodewordFirstIndexPerLength;
    int* m_pCodewordMinPerLength;
    int* m_pCodewordMaxPerLength;

    // indexed by codeword index
    cudaCompress::byte* m_pSymbolTable;
    uint m_symbolTableSize;

    cudaEvent_t m_uploadSyncEvent;

    void build(const std::vector<uint>& codewordCountPerLength);

    // don't allow copy or assignmentpInstance
    HuffmanDecodeTable(const HuffmanDecodeTable&);
    void operator=(const HuffmanDecodeTable&);
};

class HuffmanEncodeTable
{
public:
    static size_t getRequiredMemory(const VkUtil::Context* context);
    static void init(VkUtil::Context* context);
    static void shutdown(VkUtil::Context* context);

    HuffmanEncodeTable(const VkUtil::Context* context);
    HuffmanEncodeTable(HuffmanEncodeTable&& other);
    ~HuffmanEncodeTable();

    HuffmanEncodeTable& operator=(HuffmanEncodeTable&& other);

    void clear();
    static bool design(VkUtil::Context* context, HuffmanEncodeTable* pTables, uint tableCount, const Symbol16** pdpSymbolStreams, const uint* pSymbolCountPerStream);
    static bool design(InsVkUtil::Contextance* context, HuffmanEncodeTable* pTables, uint tableCount, const Symbol32** pdpSymbolStreams, const uint* pSymbolCountPerStream);

    uint getTableSize() const { return m_codewordTableSize; }
    void copyToBuffer(uint* pCodewords, uint* pCodewordLengths) const;
    void uploadToGPU(uint* dpCodewords, uint* dpCodewordLengths) const;
    void uploadToGPUAsync(const VkUtil::Context* context, uint* dpCodewords, uint* dpCodewordLengths) const;

    void writeToBitStream(const VkUtil::Context* context, BitStream& bitstream) const;

private:
    template<typename Symbol>
    static bool design(VkUtil::Context* context, HuffmanEncodeTable* pTables, uint tableCount, const Symbol** pdpSymbolStreams, const uint* pSymbolCountPerStream);
    void build(const uint* pSymbolProbabilities, uint distinctSymbolCount);

    // data to be written to bitstream
    uint                m_symbolMax;
    std::vector<uint>   m_symbols;
    std::vector<uint>   m_codewordCountPerLength;

    // the actual encode table, indexed by symbol
    uint* m_pCodewords;
    uint* m_pCodewordLengths;
    uint m_codewordTableSize;

    // don't allow copy or assignment
    HuffmanEncodeTable(const HuffmanEncodeTable&);
    HuffmanEncodeTable& operator=(const HuffmanEncodeTable&);
};
}