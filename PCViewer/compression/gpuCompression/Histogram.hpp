#pragma once

#include "../cpuCompression/global.h"

#include "../cpuCompression/EncodeCommon.h"
#include <cstring>


namespace cudaCompress {

class vkCompress::GpuInstance;

size_t histogramGetRequiredMemory(const vkCompress::GpuInstance* pInstance);
bool histogramInit(vkCompress::GpuInstance* pInstance);
bool histogramShutdown(vkCompress::GpuInstance* pInstance);

uint histogramGetElemCountIncrement();
uint histogramGetPaddedElemCount(uint elemCount);
void histogramPadData(vkCompress::GpuInstance* pInstance, ushort* dpData, uint elemCount);
void histogramPadData(vkCompress::GpuInstance* pInstance, uint*   dpData, uint elemCount);

bool histogram(vkCompress::GpuInstance* pInstance, uint* pdpHistograms[], uint histogramCount, const ushort* pdpData[], const uint* pElemCount, uint binCount);
bool histogram(vkCompress::GpuInstance* pInstance, uint* pdpHistograms[], uint histogramCount, const uint*   pdpData[], const uint* pElemCount, uint binCount);

}