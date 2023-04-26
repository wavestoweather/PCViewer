// ParallelSort.cpp
// 
// Copyright(c) 2021 Advanced Micro Devices, Inc.All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <cstdint>
#include "radix_pipeline.hpp"
#include <shader_compiler.hpp>
#include <vk_util.hpp>
#include <vma_initializers.hpp>
#include <c_file.hpp>
#include <filesystem>
#include <stager.hpp>
#include <util.hpp>

#include <numeric>
#include <random>
#include <vector>

#include "shaders.h"

using payload_none = radix_sort::gpu::payload_none;
using payload_32bit = radix_sort::gpu::payload_32bit;
using payload_64bit = radix_sort::gpu::payload_64bit;

//////////////////////////////////////////////////////////////////////////
// Helper functions FFX

// if these are adopted also the sizes have to be adopted in the shader radix_common.glsl
#define FFX_PARALLELSORT_SORT_BITS_PER_PASS		4
#define	FFX_PARALLELSORT_SORT_BIN_COUNT			(1 << FFX_PARALLELSORT_SORT_BITS_PER_PASS)
#define FFX_PARALLELSORT_ELEMENTS_PER_THREAD	4
#define FFX_PARALLELSORT_THREADGROUP_SIZE		128

struct FFX_ParallelSortCB
{
    uint32_t NumKeys;
    int32_t  NumBlocksPerThreadGroup;
    uint32_t NumThreadGroups;
    uint32_t NumThreadGroupsWithAdditionalBlocks;
    uint32_t NumReduceThreadgroupPerBin;
    uint32_t NumScanValues;
};

void FFX_ParallelSort_CalculateScratchResourceSize(uint32_t MaxNumKeys, uint32_t& ScratchBufferSize, uint32_t& ReduceScratchBufferSize)
{
    uint32_t BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
    uint32_t NumBlocks = (MaxNumKeys + BlockSize - 1) / BlockSize;
    uint32_t NumReducedBlocks = (NumBlocks + BlockSize - 1) / BlockSize;

    ScratchBufferSize = FFX_PARALLELSORT_SORT_BIN_COUNT * NumBlocks * sizeof(uint32_t);
    ReduceScratchBufferSize = FFX_PARALLELSORT_SORT_BIN_COUNT * NumReducedBlocks * sizeof(uint32_t);
}

void FFX_ParallelSort_SetConstantAndDispatchData(uint32_t NumKeys, uint32_t MaxThreadGroups, FFX_ParallelSortCB& ConstantBuffer, uint32_t& NumThreadGroupsToRun, uint32_t& NumReducedThreadGroupsToRun)
{
    ConstantBuffer.NumKeys = NumKeys;

    uint32_t BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;
    uint32_t NumBlocks = (NumKeys + BlockSize - 1) / BlockSize;

    // Figure out data distribution
    NumThreadGroupsToRun = MaxThreadGroups;
    uint32_t BlocksPerThreadGroup = (NumBlocks / NumThreadGroupsToRun);
    ConstantBuffer.NumThreadGroupsWithAdditionalBlocks = NumBlocks % NumThreadGroupsToRun;

    if (NumBlocks < NumThreadGroupsToRun)
    {
        BlocksPerThreadGroup = 1;
        NumThreadGroupsToRun = NumBlocks;
        ConstantBuffer.NumThreadGroupsWithAdditionalBlocks = 0;
    }

    ConstantBuffer.NumThreadGroups = NumThreadGroupsToRun;
    ConstantBuffer.NumBlocksPerThreadGroup = BlocksPerThreadGroup;

    // Calculate the number of thread groups to run for reduction (each thread group can process BlockSize number of entries)
    NumReducedThreadGroupsToRun = FFX_PARALLELSORT_SORT_BIN_COUNT * ((BlockSize > NumThreadGroupsToRun) ? 1 : (NumThreadGroupsToRun + BlockSize - 1) / BlockSize);
    ConstantBuffer.NumReduceThreadgroupPerBin = NumReducedThreadGroupsToRun / FFX_PARALLELSORT_SORT_BIN_COUNT;
    ConstantBuffer.NumScanValues = NumReducedThreadGroupsToRun;	// The number of reduce thread groups becomes our scan count (as each thread group writes out 1 value that needs scan prefix)
}

//////////////////////////////////////////////////////////////////////////
// Helper functions for Vulkan

// Transition barrier
inline VkBufferMemoryBarrier buffer_transition(VkBuffer buffer, VkAccessFlags before, VkAccessFlags after, size_t size)
{
    VkBufferMemoryBarrier bufferBarrier = {};
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.srcAccessMask = before;
    bufferBarrier.dstAccessMask = after;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.size = size;

    return bufferBarrier;
}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t>get_shader_code_count() {
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_none>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_count_ubyte;
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_count_ushort;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_count_uint;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_count_ulong;
    }
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_32bit>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_count_ubyte_pay; 
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_count_ushort_pay;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_count_uint_pay;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_count_ulong_pay;
    }
    // TODO 64 bit payload
    return {};
}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_reduce() {
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_none>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_reduce_ubyte;
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_reduce_ushort;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_reduce_uint;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_reduce_ulong;
    }
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_32bit>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_reduce_ubyte_pay; 
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_reduce_ushort_pay;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_reduce_uint_pay;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_reduce_ulong_pay;
    }
    // TODO 64 bit payload
    return {};
}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_scan() {
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_none>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_scan_ubyte;
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_scan_ushort;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_scan_uint;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_scan_ulong;
    }
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_32bit>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_scan_ubyte_pay; 
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_scan_ushort_pay;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_scan_uint_pay;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_scan_ulong_pay;
    }
    // TODO 64 bit payload
    return {};
}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_scan_add() {
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_none>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_scan_add_ubyte;
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_scan_add_ushort;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_scan_add_uint;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_scan_add_ulong;
    }
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_32bit>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_scan_add_ubyte_pay; 
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_scan_add_ushort_pay;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_scan_add_uint_pay;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_scan_add_ulong_pay;
    }
    // TODO 64 bit payload
    return {};
}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_scatter() {
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_none>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_scatter_ubyte;
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_scatter_ushort;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_scatter_uint;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_scatter_ulong;
    }
    if constexpr(std::is_same_v<P, radix_sort::gpu::payload_32bit>){
        if constexpr(sizeof(T) == 1) return ShaderBinaries::radix_scatter_ubyte_pay; 
        if constexpr(sizeof(T) == 2) return ShaderBinaries::radix_scatter_ushort_pay;
        if constexpr(sizeof(T) == 4) return ShaderBinaries::radix_scatter_uint_pay;
        if constexpr(sizeof(T) == 8) return ShaderBinaries::radix_scatter_ulong_pay;
    }
    // TODO 64 bit payload
    return {};
}

// returns a pair of shader codes, the first to convert to sortable, the second to convert from sortable
template<typename T>
std::pair<util::memory_view<const uint8_t>, util::memory_view<const uint8_t>> get_shader_code_convert(){ 
    if constexpr(std::is_same_v<T, int8_t>)  return {ShaderBinaries::radix_convert_byte_ubyte,   ShaderBinaries::radix_convert_ubyte_byte};
    if constexpr(std::is_same_v<T, int16_t>) return {ShaderBinaries::radix_convert_short_ushort, ShaderBinaries::radix_convert_ushort_short};
    if constexpr(std::is_same_v<T, int>)     return {ShaderBinaries::radix_convert_int_uint,     ShaderBinaries::radix_convert_uint_int};
    if constexpr(std::is_same_v<T, int64_t>) return {ShaderBinaries::radix_convert_long_ulong,   ShaderBinaries::radix_convert_ulong_long};
    if constexpr(std::is_same_v<T, float>)   return {ShaderBinaries::radix_convert_float_uint,   ShaderBinaries::radix_convert_uint_float};
    if constexpr(std::is_same_v<T, double>)  return {ShaderBinaries::radix_convert_double_ulong,ShaderBinaries::radix_convert_ulong_double};
    return {};
}

// Compile specified radix sort shader and create pipeline
template<typename T, typename P>
inline std::tuple<VkPipeline, VkPipeline, VkPipeline, VkPipeline, VkPipeline, VkPipeline, VkPipeline> create_radix_pipelines(const VkPipelineLayout pipeline_layout)
{
    VkPipeline count, reduce, scan, scan_add, scatter, convert, inv_convert;

    uint32_t local_size = FFX_PARALLELSORT_THREADGROUP_SIZE;
    auto specialization_entry = util::vk::initializers::specializationMapEntry(0, 0, sizeof(local_size));
    auto specialization_info = util::vk::initializers::specializationInfo(specialization_entry, util::memory_view(local_size));

    auto spir_v = get_shader_code_count<T, P>();
    auto shader_module = util::vk::create_scoped_shader_module(spir_v);
    auto shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
    auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    count = util::vk::create_compute_pipeline(pipeline_info);

    spir_v = get_shader_code_reduce<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    reduce = util::vk::create_compute_pipeline(pipeline_info);
    
    spir_v = get_shader_code_scan<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    scan = util::vk::create_compute_pipeline(pipeline_info);
    
    spir_v = get_shader_code_scan_add<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    scan_add = util::vk::create_compute_pipeline(pipeline_info);
    
    spir_v = get_shader_code_scatter<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    scatter = util::vk::create_compute_pipeline(pipeline_info);

    if constexpr(!std::is_unsigned_v<T>){
        auto [conv_spir_v, inv_conv_spir_v] = get_shader_code_convert<T>();
        shader_module = util::vk::create_scoped_shader_module(conv_spir_v);
        shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
        convert = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(inv_conv_spir_v);
        shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, &specialization_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
        inv_convert = util::vk::create_compute_pipeline(pipeline_info);
    }

    return {count, reduce, scan, scan_add, scatter, convert, inv_convert};
}


// Create all of the sort data for the sample
template<typename T, typename P>
radix_sort::gpu::radix_pipeline<T, P>::radix_pipeline()
{
    // Create Pipeline layout for Sorting
    // Only single layout needed, as all only use the push constants
    
    auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(push_constants), 0);
    auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant_range);
    _SortPipelineLayout = util::vk::create_pipeline_layout(layout_info);
        
    //////////////////////////////////////////////////////////////////////////
    // Create pipelines for radix sort
    // SetupIndirectParams (indirect only)
    std::tie(_FPSCountPipeline, _FPSCountReducePipeline, _FPSScanPipeline, _FPSScanAddPipeline, _FPSScatterPipeline, _FPSConvertPipeline, _FPSInvConvertPipeline) = create_radix_pipelines<T, P>(_SortPipelineLayout);

    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _fence = util::vk::create_fence(fence_info);
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
}

template<typename T, typename P>
radix_sort::gpu::radix_pipeline<T,P>& radix_sort::gpu::radix_pipeline<T,P>::instance(){
    static radix_pipeline<T,P> singleton;
    return singleton;
}

template<typename T, typename P>
typename radix_sort::gpu::radix_pipeline<T, P>::tmp_memory_info_t radix_sort::gpu::radix_pipeline<T, P>::calc_tmp_memory_info(const sort_info& info) const{
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");
    constexpr int sizeof_payload = sizeof(P);
    size_t size{};
    tmp_memory_info_t mem_info{};
    // back buffer must contain a buffer the size of the normal data plus the size of the payload data
    mem_info.back_buffer = 0;           size += info.element_count * (sizeof(T) + sizeof(P));
    mem_info.scratch_buffer = size;
    mem_info.size = size;
    return mem_info;
}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const{
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");
    assert(info.src_vk_buffer && info.back_vk_buffer && info.payload_src_vk_buffer && info.payload_back_vk_buffer && info.scratch_vk_buffer && info.scratch_reduced_vk_buffer);
    constexpr bool has_payload = !std::is_same_v<P, payload_none>;

    uint32_t num_thread_groups;
    uint32_t num_reduced_thread_groups;
    FFX_ParallelSortCB constant_buffer_data{};
    FFX_ParallelSort_SetConstantAndDispatchData(as<uint32_t>(info.element_count), _MaxNumThreadgroups, constant_buffer_data, num_thread_groups, num_reduced_thread_groups);

    // setting up indirect constant buffer
    push_constants pc{};
    pc.scan_scratch = info.scratch_buffer;
    pc.scratch_reduced = info.scratch_reduced_buffer;
    pc.num_keys = as<uint32_t>(info.element_count);
    pc.num_blocks_per_threadgroup = constant_buffer_data.NumBlocksPerThreadGroup;
    pc.num_thread_groups = constant_buffer_data.NumThreadGroups;
    pc.num_thread_groups_with_additional_blocks = constant_buffer_data.NumThreadGroupsWithAdditionalBlocks;
    pc.num_reduce_threadgroup_per_bin = constant_buffer_data.NumReduceThreadgroupPerBin;
    pc.num_scan_values = constant_buffer_data.NumScanValues;

    // converting the data to a sortable format if needed
    if(_use_convert_pipeline){
        pc.src_values = pc.dst_values = info.src_buffer;
        vkCmdPushConstants(command_buffer, _SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &pc);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSConvertPipeline);
        uint32_t dispatch_count = (pc.num_keys + FFX_PARALLELSORT_THREADGROUP_SIZE - 1) / FFX_PARALLELSORT_THREADGROUP_SIZE;
        vkCmdDispatch(command_buffer, dispatch_count, 1, 1);
    
        auto barrier = buffer_transition(info.src_vk_buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    for(int shift: util::i_range(0, as<int>(sizeof(T)) * 8, FFX_PARALLELSORT_SORT_BITS_PER_PASS)){
        const bool even_iter = ((shift / FFX_PARALLELSORT_SORT_BITS_PER_PASS) & 1) ^ 1;
        pc.bit_shift = shift;
        pc.src_values = even_iter ? info.src_buffer: info.back_buffer;
        pc.dst_values = even_iter ? info.back_buffer: info.src_buffer;
        pc.src_payload = even_iter ? info.payload_src_buffer: info.payload_back_buffer;
        pc.dst_payload = even_iter ? info.payload_back_buffer: info.payload_src_buffer;
        
        vkCmdPushConstants(command_buffer, _SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &pc);

        // sort count
        {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSCountPipeline);
            vkCmdDispatch(command_buffer, num_thread_groups, 1, 1);
            
            auto barrier = buffer_transition(info.scratch_vk_buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }
        
        // sort reduce
        {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSCountReducePipeline);
            vkCmdDispatch(command_buffer, num_reduced_thread_groups, 1, 1);
        
            auto barrier = buffer_transition(info.scratch_reduced_vk_buffer , VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }
        
        // sort scan
        {
            // scan prefix of reduced values
            // scan prefix is done from reducedscratchbuffer to reducedscratchbuffer
            // in shader these are called SumTable and ReduceTable
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSScanPipeline);
            vkCmdDispatch(command_buffer, 1, 1, 1);

            auto barrier = buffer_transition(info.scratch_reduced_vk_buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
            
            // scan prefix on the histogram with partial sums that were just now done
            // scan prefix is done with scratchbuffer on binding 1 and 2 and with reduced scratchbuffer on binding 2
            // in shader these are called ScanSrc[0] ScanDst[1] and ScanScratch[2]
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSScanAddPipeline);
            vkCmdDispatch(command_buffer, num_reduced_thread_groups, 1, 1);

            barrier = buffer_transition(info.scratch_vk_buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        // sort scatter
        {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSScatterPipeline);
            vkCmdDispatch(command_buffer, num_thread_groups, 1, 1);
        }

        // finishing everything with barriers
        std::array<VkBufferMemoryBarrier, 2> memory_barriers;
        int barrier_count{};
        auto buffer = even_iter ?
                        info.back_vk_buffer:
                        info.src_vk_buffer;
        memory_barriers[barrier_count++] = buffer_transition(buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
        if constexpr (has_payload){
            auto payload_buffer = even_iter ?
                                    info.payload_back_vk_buffer:
                                    info.payload_src_vk_buffer;
            if(payload_buffer != buffer)
                memory_barriers[barrier_count++] = buffer_transition(payload_buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
        }
        vkCmdPipelineBarrier(command_buffer,  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, barrier_count, memory_barriers.data(), 0, nullptr);
    }
    // converting the data back from a sortable format if needed
    if(_use_convert_pipeline){
        pc.src_values = pc.dst_values = info.src_buffer;
        vkCmdPushConstants(command_buffer, _SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &pc);
        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSInvConvertPipeline);
        uint32_t dispatch_count = (pc.num_keys + FFX_PARALLELSORT_THREADGROUP_SIZE - 1) / FFX_PARALLELSORT_THREADGROUP_SIZE;
        vkCmdDispatch(command_buffer, dispatch_count, 1, 1);
    
        auto barrier = buffer_transition(info.src_vk_buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_WHOLE_SIZE);
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }
}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::sort(const sort_info& info){
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");
    wait_for_fence();
    vkResetFences(globals::vk_context.device, 1, &_fence);
    if(_command_buffer)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
    _command_buffer = util::vk::create_begin_command_buffer(_command_pool);
    record_sort(_command_buffer, info);
    util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue.access().get(), {}, {}, {}, _fence);
}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::sort(const sort_info_cpu& info){
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");
    if(info.src_data.size() != info.dst_data.size() || (info.payload_src_data.size() && (info.payload_src_data.size() != info.payload_dst_data.size() || info.src_data.size() != info.payload_src_data.size()))){
        logger << logging::error_prefix << " Sizes for sorting do not align, nothing executed. Sizes are; src(" << info.src_data.size() << "), dst(" << info.dst_data.size() << "), payload_src(" << info.payload_src_data.size() << "), payload_dst(" << info.payload_dst_data.size() << ")" << logging::endl;
        return;
    }
    const size_t padded_src_size = util::align<size_t>(info.src_data.byte_size(), 4) , padded_payload_size = util::align<size_t>(info.payload_src_data.byte_size(), 4);
    // creating the required additional resources
    uint32_t scratch_buffer_size, scratch_buffer_reduced_size;
    FFX_ParallelSort_CalculateScratchResourceSize(as<uint32_t>(info.src_data.size()), scratch_buffer_size, scratch_buffer_reduced_size);

    uint32_t data_size = 2 * as<uint32_t>(padded_src_size); // front and back buffer
    data_size += 2 * as<uint32_t>(padded_payload_size);
    auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data_size + scratch_buffer_size + scratch_buffer_reduced_size);
    auto memory_info = util::vma::initializers::allocationCreateInfo();
    auto gpu_buffer = util::vk::create_buffer(buffer_info, memory_info);

    // uploading the data
    structures::stager::staging_buffer_info staging_info{};
    staging_info.data_upload = info.src_data;
    staging_info.dst_buffer = gpu_buffer.buffer;
    auto staging_begin = std::chrono::system_clock::now();
    globals::stager.add_staging_task(staging_info);
    if(info.payload_src_data.size()){
        staging_info.data_upload = info.payload_src_data;
        staging_info.dst_buffer_offset = 2 * padded_src_size;
        globals::stager.add_staging_task(staging_info);
    }

    sort_info s_info{};
    s_info.src_buffer                = util::vk::get_buffer_address(gpu_buffer);
    s_info.back_buffer               = s_info.src_buffer + padded_src_size;
    s_info.payload_src_buffer        = s_info.back_buffer + padded_src_size;
    s_info.payload_back_buffer       = s_info.payload_src_buffer + padded_payload_size;
    s_info.scratch_buffer            = s_info.payload_back_buffer + padded_payload_size;
    s_info.scratch_reduced_buffer    = s_info.scratch_buffer + scratch_buffer_size;
    s_info.element_count             = info.src_data.size();
    s_info.src_vk_buffer             = gpu_buffer.buffer;  
    s_info.back_vk_buffer            = gpu_buffer.buffer;   
    s_info.payload_src_vk_buffer     = gpu_buffer.buffer;    
    s_info.payload_back_vk_buffer    = gpu_buffer.buffer;    
    s_info.scratch_vk_buffer         = gpu_buffer.buffer;
    s_info.scratch_reduced_vk_buffer = gpu_buffer.buffer;    
    
    globals::stager.wait_for_completion();
    auto staging_end = std::chrono::system_clock::now();
    auto staging_time = std::chrono::duration<double>(staging_end - staging_begin).count();
    auto upload_size_mb = (padded_src_size + padded_payload_size) / float(1 << 20);
    if(_print_pure_gpu_times)
        logger << logging::info_prefix << " Uploading " << upload_size_mb << " MByte in " << staging_time <<" s [" << upload_size_mb / staging_time << " MBbyte/s]" << logging::endl;
    auto sort_begin = std::chrono::system_clock::now();
    sort(s_info);
    wait_for_fence();
    auto sort_end = std::chrono::system_clock::now();
    if(_print_pure_gpu_times)
        logger << logging::info_prefix << " Sorting on the gpu (without up/download) for " << info.src_data.size() << " elements took " << std::chrono::duration<double>(sort_end - sort_begin).count() << " s." << logging::endl;

    // DEBUG code
    std::vector<uint32_t> test(gpu_buffer.size / sizeof(int));

    // downloading the sorted list
    staging_info.data_upload = {};
    staging_info.transfer_dir = structures::stager::transfer_direction::download;
    //staging_info.data_download = util::memory_view(test);//info.dst_data;
    staging_info.data_download = info.dst_data;
    staging_info.dst_buffer_offset = 0;
    globals::stager.add_staging_task(staging_info);
    if(info.payload_src_data.size()){
        staging_info.data_download = info.payload_dst_data;
        staging_info.dst_buffer_offset = 2 * padded_src_size;
        globals::stager.add_staging_task(staging_info);
    }
    globals::stager.wait_for_completion();

    //std::vector<int> first_sort(test.begin() + info.src_data.size(), test.begin() + 2 * info.src_data.size());
    //std::vector<int> scratch(test.begin() + 2 * info.src_data.size(), test.begin() + 2 * info.src_data.size() + scratch_buffer_size / 4);
    //std::vector<int> scratch_reduce(test.begin() + 2 * info.src_data.size() + scratch_buffer_size / 4, test.begin() + 2 * info.src_data.size() + scratch_buffer_size / 4 + scratch_buffer_reduced_size / 4);

    // cleaning up resources
    util::vk::destroy_buffer(gpu_buffer);
}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::wait_for_fence(uint64_t timeout){
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_fence, VK_TRUE, timeout);
    util::check_vk_result(res);
}


// explicit instantiation of template functions to be able to link against them
#define inst_radix_pipeline_p(type, payload) template radix_sort::gpu::radix_pipeline<type, payload>::radix_pipeline()
#define inst_radix_pipeline(type) inst_radix_pipeline_p(type, payload_none)
inst_radix_pipeline(int8_t);
inst_radix_pipeline(uint8_t);
inst_radix_pipeline(int16_t);
inst_radix_pipeline(uint16_t);
inst_radix_pipeline(int);
inst_radix_pipeline(uint32_t);
inst_radix_pipeline(float);
inst_radix_pipeline_p(int, payload_32bit);
inst_radix_pipeline_p(uint32_t, payload_32bit);
inst_radix_pipeline_p(float, payload_32bit);

#define inst_instance_p(type, payload) template radix_sort::gpu::radix_pipeline<type, payload>& radix_sort::gpu::radix_pipeline<type, payload>::instance()
#define inst_instance(type) inst_instance_p(type, payload_none)
inst_instance(int8_t);
inst_instance(uint8_t);
inst_instance(int16_t);
inst_instance(uint16_t);
inst_instance(int);
inst_instance(uint32_t);
inst_instance(float);
inst_instance_p(int, payload_32bit);
inst_instance_p(uint32_t, payload_32bit);
inst_instance_p(float, payload_32bit);

#define inst_calc_tmp_mem_info_p(type, payload) template radix_sort::gpu::radix_pipeline<type, payload>::tmp_memory_info_t radix_sort::gpu::radix_pipeline<type, payload>::calc_tmp_memory_info(const sort_info& info) const
#define inst_calc_tmp_mem_info(type) inst_calc_tmp_mem_info_p(type, payload_none)
inst_calc_tmp_mem_info(int8_t);
inst_calc_tmp_mem_info(uint8_t);
inst_calc_tmp_mem_info(int16_t);
inst_calc_tmp_mem_info(uint16_t);
inst_calc_tmp_mem_info(int);
inst_calc_tmp_mem_info(uint32_t);
inst_calc_tmp_mem_info(float);
inst_calc_tmp_mem_info_p(int, payload_32bit);
inst_calc_tmp_mem_info_p(uint32_t, payload_32bit);
inst_calc_tmp_mem_info_p(float, payload_32bit);

#define inst_record_sort_p(type, payload) template void radix_sort::gpu::radix_pipeline<type, payload>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const
#define inst_record_sort(type) inst_record_sort_p(type, payload_none)
inst_record_sort(int8_t);
inst_record_sort(uint8_t);
inst_record_sort(int16_t);
inst_record_sort(uint16_t);
inst_record_sort(int);
inst_record_sort(uint32_t);
inst_record_sort(float);
inst_record_sort_p(int, payload_32bit);
inst_record_sort_p(uint32_t, payload_32bit);
inst_record_sort_p(float, payload_32bit);

#define inst_sort_p(type, payload) template void radix_sort::gpu::radix_pipeline<type, payload>::sort(const sort_info& info)
#define inst_sort(type) inst_sort_p(type, payload_none)
inst_sort(int8_t);
inst_sort(uint8_t);
inst_sort(int16_t);
inst_sort(uint16_t);
inst_sort(int);
inst_sort(uint32_t);
inst_sort(float);
inst_sort_p(int, payload_32bit);
inst_sort_p(uint32_t, payload_32bit);
inst_sort_p(float, payload_32bit);

#define inst_sort_cpu_p(type, payload) template void radix_sort::gpu::radix_pipeline<type, payload>::sort(const sort_info_cpu& info)
#define inst_sort_cpu(type) inst_sort_cpu_p(type, payload_none)
inst_sort_cpu(int8_t);
inst_sort_cpu(uint8_t);
inst_sort_cpu(int16_t);
inst_sort_cpu(uint16_t);
inst_sort_cpu(int);
inst_sort_cpu(uint32_t);
inst_sort_cpu(float);
inst_sort_cpu_p(int, payload_32bit);
inst_sort_cpu_p(uint32_t, payload_32bit);
inst_sort_cpu_p(float, payload_32bit);

#define inst_wait_for_fence_p(type, payload) template void radix_sort::gpu::radix_pipeline<type, payload>::wait_for_fence(uint64_t timeout)
#define inst_wait_for_fence(type) inst_wait_for_fence_p(type, payload_none)
inst_wait_for_fence(int8_t);
inst_wait_for_fence(uint8_t);
inst_wait_for_fence(int16_t);
inst_wait_for_fence(uint16_t);
inst_wait_for_fence(int);
inst_wait_for_fence(uint32_t);
inst_wait_for_fence(float);
inst_wait_for_fence_p(int, payload_32bit);
inst_wait_for_fence_p(uint32_t, payload_32bit);
inst_wait_for_fence_p(float, payload_32bit);
