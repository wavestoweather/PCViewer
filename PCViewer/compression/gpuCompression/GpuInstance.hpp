#pragma once

#include "../../VkUtil.h"

struct GpuInstance{
    VkUtil::Context vkContext{};      // holds gpu device information

    uint m_streamCountMax{};
    uint m_elemCountPerStreamMax{};
    uint m_codingBlockSize{};
    uint m_log2HuffmanDistinctSymbolCountMax{14};
    // todo fill
};