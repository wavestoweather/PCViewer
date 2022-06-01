#pragma once

#include "../../VkUtil.h"

namespace vkCompress{
struct GpuInstance{
    VkUtil::Context vkContext{};      // holds gpu device information

    uint m_streamCountMax{};
    uint m_elemCountPerStreamMax{};
    uint m_codingBlockSize{};
    uint m_log2HuffmanDistinctSymbolCountMax{14};
    // todo fill

    struct RunLengthResources
    {
        uint* pReadback;
        std::vector<void*> syncEventsReadback;

        byte* pUpload;
        void* syncEventUpload;

        RunLengthResources()
            : pReadback(nullptr), pUpload(nullptr), syncEventUpload(0) {}
    } RunLength;
};
}