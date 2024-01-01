#pragma once

#include "../cpuCompression/global.h"

#include "../cpuCompression/EncodeCommon.h"
#include <cstring>


namespace vkCompress {

struct GpuInstance;


size_t histogramGetRequiredMemory(const GpuInstance* pInstance);
bool histogramInit(GpuInstance* pInstance);
bool histogramShutdown(GpuInstance* pInstance);

uint histogramGetElemCountIncrement();
uint histogramGetPaddedElemCount(uint elemCount);
void histogramPadData(GpuInstance* pInstance, ushort* dpData, uint elemCount);
void histogramPadData(GpuInstance* pInstance, uint*   dpData, uint elemCount);

bool histogram(GpuInstance* pInstance, uint* pdpHistograms[], uint histogramCount, const ushort* pdpData[], const uint* pElemCount, uint binCount);
bool histogram(GpuInstance* pInstance, uint* pdpHistograms[], uint histogramCount, const uint*   pdpData[], const uint* pElemCount, uint binCount);

}