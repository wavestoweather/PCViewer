#pragma once

#include "../cpuCompression/util.h"

#include <vector>

#include "../cpuCompression/EncodeCommon.h"


namespace vkCompress {

class GpuInstance;


size_t runLengthGetRequiredMemory(const GpuInstance* pInstance);
bool runLengthInit(GpuInstance* pInstance);
bool runLengthShutdown(GpuInstance* pInstance);

// note: for decode, output arrays (pdpSymbols) must be zeroed!

bool runLengthEncode(GpuInstance* pInstance, Symbol16** pdpSymbolsCompact, Symbol16** pdpZeroCounts, const Symbol16** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact);
bool runLengthDecode(GpuInstance* pInstance, const Symbol16** pdpSymbolsCompact, const Symbol16** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol16** pdpSymbols, const uint* pSymbolCount, uint streamCount);
bool runLengthDecode(GpuInstance* pInstance, const Symbol16* dpSymbolsCompact, const Symbol16* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol16** pdpSymbols, uint symbolCount, uint streamCount);
bool runLengthDecode(GpuInstance* pInstance, VkCommandBuffer commands, VkDescriptorSet rLSet);

bool runLengthEncode(GpuInstance* pInstance, Symbol32** pdpSymbolsCompact, Symbol32** pdpZeroCounts, const Symbol32** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact);
bool runLengthDecode(GpuInstance* pInstance, const Symbol32** pdpSymbolsCompact, const Symbol32** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol32** pdpSymbols, const uint* pSymbolCount, uint streamCount);
bool runLengthDecode(GpuInstance* pInstance, const Symbol32* dpSymbolsCompact, const Symbol32* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol32** pdpSymbols, uint symbolCount, uint streamCount);


}