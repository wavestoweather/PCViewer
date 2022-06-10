#pragma once

#include "../../VkUtil.h"

namespace vkCompress{
struct GpuInstance{
public:
    GpuInstance(VkUtil::Context context);
    ~GpuInstance();

    VkUtil::Context vkContext{};      // holds gpu device information

    uint m_streamCountMax{};
    uint m_elemCountPerStreamMax{};
    uint m_codingBlockSize{};
    uint m_log2HuffmanDistinctSymbolCountMax{14};
    // todo fill

    struct HistogramResources
    {
        byte* pUpload{};
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
};
}