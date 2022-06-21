#pragma once

#include "../../VkUtil.h"
#include "../cpuCompression/global.h"
#include "ScanPlan.hpp"
#include "ReducePlan.hpp"
#include "HuffmanTable.h"

namespace vkCompress{
struct HuffmanGPUStreamInfo;

struct GpuInstance{
public:
    GpuInstance(VkUtil::Context context, uint32_t streamCountMax, uint32_t elemCountPerStreamMax, uint32_t codingBlockSize, uint32_t log2HuffmanDistinctSymbolCountMax);
    ~GpuInstance();

    VkUtil::Context vkContext{};      // holds gpu device information

    uint m_streamCountMax{1};
    uint m_elemCountPerStreamMax{};
    uint m_codingBlockSize{};
    uint m_log2HuffmanDistinctSymbolCountMax{14};
    uint m_warpSize{};

    ScanPlan* m_pScanPlan{};
    ReducePlan* m_pReducePlan{};
    // todo fill

    // TIER 1
    struct EncodeResources
    {
        // encode*:
        // used for downloads
        uint* pCodewordBuffer;
        uint* pOffsetBuffer;
        uint* pEncodeCodewords;
        uint* pEncodeCodewordLengths;

        HuffmanGPUStreamInfo* pEncodeSymbolStreamInfos;

        std::vector<HuffmanEncodeTable> symbolEncodeTables;
        std::vector<HuffmanEncodeTable> zeroCountEncodeTables;

        VkFence encodeFinishedFence;

        // decode*:
        // decode resources are multi-buffered to avoid having to sync too often
        struct DecodeResources
        {
            VkFence syncFence;

            std::vector<HuffmanDecodeTable> symbolDecodeTables;
            std::vector<HuffmanDecodeTable> zeroCountDecodeTables;
            VkBuffer pSymbolDecodeTablesBuffer;
            VkDeviceSize symbolDecodeTablesBufferOffset;
            VkBuffer pZeroCountDecodeTablesBuffer;
            VkDeviceSize ZeroCountDecodeTablesBufferOffset;

            VkBuffer  pCodewordStreams;
            VkDeviceSize codewordStreamsOffset;
            VkBuffer  pSymbolOffsets;
            VkDeviceSize symbolOffsetsOffset;
            VkBuffer  pZeroCountOffsets;
            VkDeviceSize zeroCountOffsetsOffset;

            // actually only these have to be filled for the decoding to work
            // so: Create in the decode function buffers and fill them with the correct infos,
            //  and then bind them to the buffer adresses here
            // for the huffman decode function: create a vulkan buffer out of this array and bind that to the pipeline
            HuffmanGPUStreamInfo* pSymbolStreamInfos;
            HuffmanGPUStreamInfo* pZeroCountStreamInfos;

            VkDeviceMemory memory;

            DecodeResources()
                : syncFence(0)
                , pSymbolDecodeTablesBuffer(nullptr), pZeroCountDecodeTablesBuffer(nullptr)
                , pCodewordStreams(nullptr), pSymbolOffsets(nullptr), pZeroCountOffsets(nullptr)
                , pSymbolStreamInfos(nullptr), pZeroCountStreamInfos(nullptr) {}
        };
        const static int ms_decodeResourcesCount = 8;
        DecodeResources Decode[ms_decodeResourcesCount];
        int nextDecodeResources;
        DecodeResources& GetDecodeResources() {
            DecodeResources& result = Decode[nextDecodeResources];
            nextDecodeResources = (nextDecodeResources + 1) % ms_decodeResourcesCount;
            return result;
        }


        //util::CudaTimerResources timerEncodeLowDetail;
        //util::CudaTimerResources timerEncodeHighDetail;
        //util::CudaTimerResources timerDecodeLowDetail;
        //util::CudaTimerResources timerDecodeHighDetail;

        EncodeResources()
            : pCodewordBuffer(nullptr), pOffsetBuffer(nullptr)
            , pEncodeCodewords(nullptr), pEncodeCodewordLengths(nullptr)
            , pEncodeSymbolStreamInfos(nullptr)
            , encodeFinishedFence(0)
            , nextDecodeResources(0) {}
    } Encode;

    struct HistogramResources
    {
        //byte* pUpload{};
        VkBuffer pUpload{};
        VkDeviceMemory pUploadMem{};
        VkFence syncFence{};
        VkUtil::PipelineInfo pipelineInfo{};
    } Histogram;

    struct HuffmanTableResources
    {
        uint* pReadback{};
        VkUtil::PipelineInfo pipelineInfo{};
    } HuffmanTable;

    struct RunLengthResources
    {
        uint* pReadback{};
        std::vector<void*> syncEventsReadback;

        byte* pUpload{};
        VkFence syncFenceUpload{};
        VkUtil::PipelineInfo pipelineInfo{};
    } RunLength;

    struct DWTResources
    {
        VkUtil::PipelineInfo pipelineInfo{};
    } DWT;

    struct QuantizationResources
    {
        VkUtil::PipelineInfo pipelineInfo{};
    } Quantization;

private:
    uint m_bufferSize;
};
}