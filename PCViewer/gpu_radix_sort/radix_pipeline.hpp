
// Copywrite Josef Stumpfegger 2023

#pragma once
#include <vk_context.hpp>

namespace radix_sort{namespace gpu{
struct payload_none{};
struct payload_32bit{};
struct payload_64bit{};
template<typename T, typename P = payload_none>
class radix_pipeline{
    // instead of binding the buffers via descriptor set we use a
    // push constant with buffer device addresses, which allows us to avoid descriptor sets
    struct push_constants{
    uint64_t src_values;
    uint64_t dst_values;
    uint64_t src_payload;
    uint64_t dst_payload;
    uint64_t sum_table;
    uint64_t reduce_table;
    uint64_t scan_src;
    uint64_t scan_dst;
    uint64_t scan_scratch;
    
    uint32_t bit_shift;
    uint32_t num_keys;					
    uint32_t num_blocks_per_threadgroup;
    uint32_t num_thread_groups;
    uint32_t num_thread_groups_with_additional_blocks;
    uint32_t num_reduce_threadgroup_per_bin;
    uint32_t num_scan_values;
    };

    // Sample resources
    structures::buffer_info _SrcKeyBuffers[3];     // 32 bit source key buffers (for 1080, 2K, 4K resolution)
    structures::buffer_info _SrcPayloadBuffers;    // 32 bit source payload buffers
    structures::buffer_info _DstKeyBuffers[2];     // 32 bit destination key buffers (when not doing in place writes)
    structures::buffer_info _DstPayloadBuffers[2]; // 32 bit destination payload buffers (when not doing in place writes)
    structures::buffer_info _FPSScratchBuffer;             // Sort scratch buffer
    structures::buffer_info _FPSReducedScratchBuffer;      // Sort reduced scratch buffer

    VkPipelineLayout        _SortPipelineLayout;

    VkPipeline              _FPSCountPipeline;
    VkPipeline              _FPSCountReducePipeline;
    VkPipeline              _FPSScanPipeline;
    VkPipeline              _FPSScanAddPipeline;
    VkPipeline              _FPSScatterPipeline;
    VkPipeline              _FPSScatterPayloadPipeline;
    
    VkFence                 _fence{};
    VkCommandPool           _command_pool{};
    VkCommandBuffer         _command_buffer{};

    uint32_t                _MaxNumThreadgroups{800};

    radix_pipeline();
public:
    struct tmp_memory_info_t{
        size_t
        back_buffer,
        scratch_buffer,
        size;
    };
    struct storage_flags{
        structures::buffer_info buffer;
        bool                    scratch_buffer: 1;
        bool                    reduced_scratch_buffer: 1;
        bool                    src_buffer: 1;
        bool                    back_buffer: 1;
        bool                    payload_src_buffer: 1;
        bool                    payload_back_buffer: 1;
    };
    struct sort_info{
        VkDeviceAddress                     src_buffer;
        VkDeviceAddress                     back_buffer;
        VkDeviceAddress                     payload_src_buffer;
        VkDeviceAddress                     payload_back_buffer;
        VkDeviceAddress                     scratch_buffer;
        VkDeviceAddress                     scratch_reduced_buffer;
        util::memory_view<storage_flags>    storage_buffers;    // contains the vulkan buffer of src, dst and payload, buffer layout is ignored
        size_t                              element_count;
    };
    struct sort_info_cpu{
        util::memory_view<const T> src_data;
        util::memory_view<T>       dst_data;
        util::memory_view<const P> payload_src_data;
        util::memory_view<P>       payload_dst_data;
    };

    radix_pipeline(const radix_pipeline&) = delete;
    radix_pipeline& operator=(const radix_pipeline&) = delete;

    static radix_pipeline& instance();
 
    tmp_memory_info_t calc_tmp_memory_info(const sort_info& info) const;

    // Records all pipeline commands to command_buffer.
    // No execution is performed and all temporary buffers have to be allocated
    // The result will always be placed in the src buffers
    void record_sort(VkCommandBuffer command_buffer, const sort_info& info) const;

    // Creates a command buffer and executes the sorting immediately
    // All temporary buffers have to be allocated beforehand
    // Unordered array as well as sorted array are on the gpu
    // Does not block, however next call to sort will wait for previous sort to be finished
    // The result will always be placed in the src buffers
    void sort(const sort_info& info);

    // Takes cpu data and performs the sorting on the gpu
    // blocks until sort is done
    void sort(const sort_info_cpu& info);

    // waiting for the end of a sort task submitted via sort(const sort_info& info)
    void wait_for_fence(uint64_t timeout = std::numeric_limits<uint64_t>::max());
};
}}
