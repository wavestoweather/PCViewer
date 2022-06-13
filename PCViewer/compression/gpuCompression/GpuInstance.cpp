#include "GpuInstance.hpp"
#include "../cpuCompression/util.h"

namespace vkCompress
{
    using namespace cudaCompress;
    GpuInstance::GpuInstance(VkUtil::Context context, uint32_t streamCountMax, uint32_t elemCountPerStreamMax, uint32_t codingBlockSize, uint32_t log2HuffmanDistinctSymbolCountMax):
    vkContext(context),
    m_streamCountMax(streamCountMax),
    m_elemCountPerStreamMax(elemCountPerStreamMax),
    m_codingBlockSize(codingBlockSize == 0 ? 24 : codingBlockSize), // default to 128
    m_log2HuffmanDistinctSymbolCountMax(log2HuffmanDistinctSymbolCountMax == 0 ? 14 : log2HuffmanDistinctSymbolCountMax)  // default to 14 bits (which was used before this was configurable)
    {
        if(m_log2HuffmanDistinctSymbolCountMax > 24) 
            throw std::runtime_error{"WARNING: log2HuffmanDistinctSymbolCountMax must be <= 24 (provided:" + std::to_string(m_log2HuffmanDistinctSymbolCountMax) + ")\n"};

        uint offsetCountMax = (m_elemCountPerStreamMax + m_codingBlockSize - 1) / m_codingBlockSize;

        uint rowPitch = (uint) getAlignedSize(m_elemCountPerStreamMax + 1, 128 / sizeof(uint));
        m_pScanPlan = new ScanPlan(context, sizeof(uint), m_elemCountPerStreamMax + 1, m_streamCountMax, rowPitch); // "+ 1" for total
        m_pReducePlan = new ReducePlan(sizeof(uint), m_elemCountPerStreamMax);


        size_t sizeTier0 = 0;
        sizeTier0 = max(sizeTier0, runLengthGetRequiredMemory(this));
        sizeTier0 = max(sizeTier0, huffmanGetRequiredMemory(this));
        // HuffmanEncodeTable uses histogram...
        sizeTier0 = max(sizeTier0, HuffmanEncodeTable::getRequiredMemory(this) + histogramGetRequiredMemory(this));
        sizeTier0 = max(sizeTier0, packIncGetRequiredMemory(this));
        size_t sizeTier1 = 0;
        sizeTier1 = max(sizeTier1, encodeGetRequiredMemory(this));

        m_bufferSize = sizeTier0 + sizeTier1;
        // creating all pipelines

        // Huffman table pipelines ---------------------------------------------------
        
    }
    
    GpuInstance::~GpuInstance() 
    {
        Histogram.pipelineInfo.vkDestroy(vkContext);
        HuffmanTable.pipelineInfo.vkDestroy(vkContext);
        RunLength.pipelineInfo.vkDestroy(vkContext);
        DWT.pipelineInfo.vkDestroy(vkContext);
        Quantization.pipelineInfo.vkDestroy(vkContext);
    }
}