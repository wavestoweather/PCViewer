
// Copywrite Josef Stumpfegger 2023

#pragma once
#include <vk_context.hpp>

namespace radix_sort{namespace gpu{
class radix_pipeline{
    // instead of binding the buffers via descriptor set we use a
    // push constant with buffer device addresses, which allows us to avoid descriptor sets
    struct push_constants{

    };

    // Sample resources
    structures::buffer_info _SrcKeyBuffers[3];     // 32 bit source key buffers (for 1080, 2K, 4K resolution)
    structures::buffer_info _SrcPayloadBuffers;    // 32 bit source payload buffers
    structures::buffer_info _DstKeyBuffers[2];     // 32 bit destination key buffers (when not doing in place writes)
    structures::buffer_info _DstPayloadBuffers[2]; // 32 bit destination payload buffers (when not doing in place writes)
    structures::buffer_info _FPSScratchBuffer;             // Sort scratch buffer
    structures::buffer_info _FPSReducedScratchBuffer;      // Sort reduced scratch buffer

    VkDescriptorSetLayout   _SortDescriptorSetLayoutConstants;
    VkDescriptorSet         _SortDescriptorSetConstants[3];
    VkDescriptorSetLayout   _SortDescriptorSetLayoutConstantsIndirect;
    VkDescriptorSet         _SortDescriptorSetConstantsIndirect[3];

    VkDescriptorSetLayout   _SortDescriptorSetLayoutInputOutputs;
    VkDescriptorSetLayout   _SortDescriptorSetLayoutScan;
    VkDescriptorSetLayout   _SortDescriptorSetLayoutScratch;
    VkDescriptorSetLayout   _SortDescriptorSetLayoutIndirect;

    VkDescriptorSet         _SortDescriptorSetInputOutput[2];
    VkDescriptorSet         _SortDescriptorSetScanSets[2];
    VkDescriptorSet         _SortDescriptorSetScratch;
    VkDescriptorSet         _SortDescriptorSetIndirect;
    VkPipelineLayout        _SortPipelineLayout;

    VkPipeline              _FPSCountPipeline;
    VkPipeline              _FPSCountReducePipeline;
    VkPipeline              _FPSScanPipeline;
    VkPipeline              _FPSScanAddPipeline;
    VkPipeline              _FPSScatterPipeline;
    VkPipeline              _FPSScatterPayloadPipeline;

    // Resources for indirect execution of algorithm
    structures::buffer_info _IndirectKeyCounts;            // Buffer to hold num keys for indirect dispatch
    structures::buffer_info _IndirectConstantBuffer;       // Buffer to hold radix sort constant buffer data for indirect dispatch
    structures::buffer_info _IndirectCountScatterArgs;     // Buffer to hold dispatch arguments used for Count/Scatter parts of the algorithm
    structures::buffer_info _IndirectReduceScanArgs;       // Buffer to hold dispatch arguments used for Reduce/Scan parts of the algorithm
        
    VkPipeline              _FPSIndirectSetupParametersPipeline;

    VkFence                 _fence{};
    VkCommandPool           _command_pool{};
    VkCommandBuffer         _command_buffer{};

    uint32_t                _MaxNumThreadgroups{};

    radix_pipeline();
public:
    struct sort_info{

    };
    struct sort_info_cpu{

    };

    radix_pipeline(const radix_pipeline&) = delete;
    radix_pipeline& operator=(const radix_pipeline&) = delete;

    static radix_pipeline& instance();

    void record_sort(const sort_info& info);
    void sort(const sort_info& info);
    void sort(sort_info_cpu& info);
};
}}
