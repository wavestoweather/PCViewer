#pragma once

#include "../cpuCompression/global.h"
#include <cstring>


namespace vkCompress {

class GpuInstance;

size_t packIncGetRequiredMemory(const GpuInstance* pInstance);
bool packIncInit(GpuInstance* pInstance);
bool packIncShutdown(GpuInstance* pInstance);

// dpValues must be monotonically increasing for packInc
// dpValues and dpPackedValueIncrements may be the same
bool packInc(GpuInstance* pInstance, const uint* dpValues, uint* dpPackedValueIncrements, uint valueCount, uint& bitsPerValue);
bool unpackInc(GpuInstance* pInstance, uint* dpValues, const uint* dpPackedValueIncrements, uint valueCount, uint bitsPerValue);

// simple versions that always use 16 bit per value increment
bool packInc16(GpuInstance* pInstance, const uint* dpValues, ushort* dpValueIncrements, uint valueCount);
bool unpackInc16(GpuInstance* pInstance, uint* dpValues, const ushort* dpValueIncrements, uint valueCount);

void packInc16CPU(const uint* pValues, ushort* pValueIncrements, uint valueCount); // works in-place
void unpackInc16CPU(uint* pValues, const ushort* pValueIncrements, uint valueCount); // does *not* work in-place

}