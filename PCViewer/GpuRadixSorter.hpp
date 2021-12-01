#pragma once
#include "VkUtil.h"

// Gpu accelerated radix sorting using the paper "A Memory Bandwidth-Efficient Hybrid Radix Sort on GPUs" https://arxiv.org/pdf/1611.01137.pdf
// as an implementation reference
// Provides index list sorting according to floating point data
// Floating point data is converted to hirarchical uints using http://stereopsis.com/radix.html
// For local sorting bitonic sort is used. An example for an efficient Gpu implementation: https://people.scs.carleton.ca/~dehne/teaching/4009/lectures/pdf/05c-Parallel_bitonic_sort.pdf

#define ENABLE_TEST_SORT

class GpuRadixSorter{
public:
    GpuRadixSorter(const VkUtil::Context& context);
    ~GpuRadixSorter();

    void sortUints(std::vector<uint32_t>& v);
    void sortFloats(std::vector<float>& v);

    #ifdef ENABLE_TEST_SORT
    bool checkLocalSort();  //checks local sort on an array of uints the size of 
    bool checkHistogram();  //checks histogram for a random uint array
    #endif
private:
    std::string _shaderPath = "shader/radixSort.comp.spv";
    std::string _shaderHistogramPath = "shader/radixHistogram.comp.spv";
    std::string _shaderScatteringPath = "shader/radixScattering.comp.spv";
    std::string _shaderControlPath = "shader/radixControl.comp.spv";
    std::string _shaderLocalSortPath = "shader/radixLocalSort.comp.spv";

    uint32_t _localSize = 256;
    VkUtil::Context _vkContext;
    VkUtil::PipelineInfo _pipelineInfo;
    VkUtil::PipelineInfo _localSortPipeline;
    VkUtil::PipelineInfo _histogramPipeline;
};