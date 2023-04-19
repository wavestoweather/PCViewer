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

#define FFX_CPP
#include <cstdint>
#include "FFX_ParallelSort.h"
#include "radix_pipeline.hpp"
#include <shader_compiler.hpp>
#include <vk_util.hpp>
#include <c_file.hpp>
#include <filesystem>

#include <numeric>
#include <random>
#include <vector>

#include "shaders.h"

using payload_32bit = radix_sort::gpu::payload_32bit;
using payload_64bit = radix_sort::gpu::payload_64bit;
//////////////////////////////////////////////////////////////////////////
// Helper functions for Vulkan

// Transition barrier
inline VkBufferMemoryBarrier buffer_transition(VkBuffer buffer, VkAccessFlags before, VkAccessFlags after, uint32_t size)
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
util::memory_view<const uint8_t>get_shader_code_count() {return {};}
template<> util::memory_view<const uint8_t> get_shader_code_count<int>(){return ShaderBinaries::radix_count_int;}
template<> util::memory_view<const uint8_t> get_shader_code_count<uint32_t>(){return ShaderBinaries::radix_count_uint;}
template<> util::memory_view<const uint8_t> get_shader_code_count<float>(){return ShaderBinaries::radix_count_float;}
template<> util::memory_view<const uint8_t> get_shader_code_count<float, payload_32bit>(){return ShaderBinaries::radix_count_float_32;}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_reduce() {return {};}
template<> util::memory_view<const uint8_t> get_shader_code_reduce<int>(){return ShaderBinaries::radix_reduce_int;}
template<> util::memory_view<const uint8_t> get_shader_code_reduce<uint32_t>(){return ShaderBinaries::radix_reduce_uint;}
template<> util::memory_view<const uint8_t> get_shader_code_reduce<float>(){return ShaderBinaries::radix_reduce_float;}
template<> util::memory_view<const uint8_t> get_shader_code_reduce<float, payload_32bit>(){return ShaderBinaries::radix_reduce_float_32;}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_scan() {return {};}
template<> util::memory_view<const uint8_t> get_shader_code_scan<int>(){return ShaderBinaries::radix_scan_int;}
template<> util::memory_view<const uint8_t> get_shader_code_scan<uint32_t>(){return ShaderBinaries::radix_scan_uint;}
template<> util::memory_view<const uint8_t> get_shader_code_scan<float>(){return ShaderBinaries::radix_scan_float;}
template<> util::memory_view<const uint8_t> get_shader_code_scan<float, payload_32bit>(){return ShaderBinaries::radix_scan_float_32;}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_scan_add() {return {};}
template<> util::memory_view<const uint8_t> get_shader_code_scan_add<int>(){return ShaderBinaries::radix_scan_add_int;}
template<> util::memory_view<const uint8_t> get_shader_code_scan_add<uint32_t>(){return ShaderBinaries::radix_scan_add_uint;}
template<> util::memory_view<const uint8_t> get_shader_code_scan_add<float>(){return ShaderBinaries::radix_scan_add_float;}
template<> util::memory_view<const uint8_t> get_shader_code_scan_add<float, payload_32bit>(){return ShaderBinaries::radix_scan_add_float_32;}

template<typename T, typename P = radix_sort::gpu::payload_none>
util::memory_view<const uint8_t> get_shader_code_scatter() {return {};}
template<> util::memory_view<const uint8_t> get_shader_code_scatter<int>(){return ShaderBinaries::radix_scatter_int;}
template<> util::memory_view<const uint8_t> get_shader_code_scatter<uint32_t>(){return ShaderBinaries::radix_scatter_uint;}
template<> util::memory_view<const uint8_t> get_shader_code_scatter<float>(){return ShaderBinaries::radix_scatter_float;}
template<> util::memory_view<const uint8_t> get_shader_code_scatter<float, payload_32bit>(){return ShaderBinaries::radix_scatter_float_32;}

// Compile specified radix sort shader and create pipeline
template<typename T, typename P>
inline std::tuple<VkPipeline, VkPipeline, VkPipeline, VkPipeline, VkPipeline> create_radix_pipelines(const VkPipelineLayout pipeline_layout)
{
    VkPipeline count, reduce, scan, scan_add, scatter;

    auto spir_v = get_shader_code_count<T, P>();
    auto shader_module = util::vk::create_scoped_shader_module(spir_v);
    auto shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
    auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    count = util::vk::create_compute_pipeline(pipeline_info);

    spir_v = get_shader_code_reduce<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    reduce = util::vk::create_compute_pipeline(pipeline_info);
    
    spir_v = get_shader_code_scan<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    scan = util::vk::create_compute_pipeline(pipeline_info);
    
    spir_v = get_shader_code_scan_add<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    scan_add = util::vk::create_compute_pipeline(pipeline_info);
    
    spir_v = get_shader_code_scatter<T, P>();
    shader_module = util::vk::create_scoped_shader_module(spir_v);
    shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
    pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    scatter = util::vk::create_compute_pipeline(pipeline_info);

    return {count, reduce, scan, scan_add, scatter};
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
    std::tie(_FPSCountPipeline, _FPSCountReducePipeline, _FPSScanPipeline, _FPSScanAddPipeline, _FPSScatterPipeline) = create_radix_pipelines<T, P>(_SortPipelineLayout);
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
    mem_info.scratch_buffer = size; // TODO: add scratch buffer size
    mem_info.size = size;
    return mem_info;
}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const{
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");
    constexpr bool has_payload = !std::is_same_v<P, payload_none>;

    uint32_t num_thread_groups;
    uint32_t num_reduced_thread_groups;
    FFX_ParallelSortCB constant_buffer_data{};
    FFX_ParallelSort_SetConstantAndDispatchData(info.element_count, _MaxNumThreadgroups, constant_buffer_data, num_thread_groups, num_reduced_thread_groups);

    // setting up indirect constant buffer
    push_constants pc{};
    pc.num_keys = info.element_count;
    pc.scan_scratch = info.scratch_buffer;

    for(int shift: util::i_range(0, as<int>(sizeof(T)) * 8, FFX_PARALLELSORT_SORT_BITS_PER_PASS)){
        const bool even_iter = (shift / FFX_PARALLELSORT_SORT_BIN_COUNT) & 1 ^ 1;
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
            
            auto buffer = std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.scratch_buffer;});
            auto barrier = buffer_transition(buffer->buffer.buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, buffer->buffer.size);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        // sort reduce
        {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSCountReducePipeline);
            vkCmdDispatch(command_buffer, num_reduced_thread_groups, 1, 1);
        
            auto buffer = std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.reduced_scratch_buffer;});
            auto barrier = buffer_transition(buffer->buffer.buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, buffer->buffer.size);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        // sort scan
        {
            // scan prefix of reduced values
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSScanPipeline);
            vkCmdDispatch(command_buffer, 1, 1, 1);

            auto buffer = std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.reduced_scratch_buffer;});
            auto barrier = buffer_transition(buffer->buffer.buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, buffer->buffer.size);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
            
            // scan prefix on the histogram with partial sums that were just now done
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _FPSScanAddPipeline);
            vkCmdDispatch(command_buffer, num_reduced_thread_groups, 1, 1);

            buffer = std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.scratch_buffer;});
            barrier = buffer_transition(buffer->buffer.buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, buffer->buffer.size);
            vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        // sort scatter
        {
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, has_payload ? _FPSScatterPayloadPipeline: _FPSScatterPipeline);
            vkCmdDispatch(command_buffer, num_thread_groups, 1, 1);
        }

        // finishing everything with barriers
        std::array<VkBufferMemoryBarrier, 2> memory_barriers;
        int barrier_count{};
        auto buffer = even_iter ?
                        std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.back_buffer;}):
                        std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.src_buffer;});
        memory_barriers[barrier_count++] = buffer_transition(buffer->buffer.buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, buffer->buffer.size);
        if constexpr (has_payload){
            auto payload_buffer = even_iter ?
                                    std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.payload_back_buffer;}):
                                    std::find_if(info.storage_buffers.begin(), info.storage_buffers.end(), [](const auto& b){return b.payload_src_buffer;});
            if(payload_buffer != buffer)
                memory_barriers[barrier_count++] = buffer_transition(payload_buffer->buffer.buffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, payload_buffer->buffer.size);
        }
        vkCmdPipelineBarrier(command_buffer,  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, barrier_count, memory_barriers.data(), 0, nullptr);
    }
}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::sort(const sort_info& info){
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");

}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::sort(const sort_info_cpu& info){
    static_assert(sizeof(T) <= 8 && sizeof(P) <= 8 && "Only at max 64 bit values are allowed for key and payload. If payload is larger, use index for payload and after sort map index to payload");

}

template<typename T, typename P>
void radix_sort::gpu::radix_pipeline<T, P>::wait_for_fence(uint64_t timeout){
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_fence, VK_TRUE, timeout);
    util::check_vk_result(res);
}


// explicit instantiation of template functions
template radix_sort::gpu::radix_pipeline<int>::radix_pipeline();
template radix_sort::gpu::radix_pipeline<uint32_t>::radix_pipeline();
template radix_sort::gpu::radix_pipeline<float>::radix_pipeline();
template radix_sort::gpu::radix_pipeline<float, payload_32bit>::radix_pipeline();

template radix_sort::gpu::radix_pipeline<int>& radix_sort::gpu::radix_pipeline<int>::instance();
template radix_sort::gpu::radix_pipeline<uint32_t>& radix_sort::gpu::radix_pipeline<uint32_t>::instance();
template radix_sort::gpu::radix_pipeline<float>& radix_sort::gpu::radix_pipeline<float>::instance();
template radix_sort::gpu::radix_pipeline<float, payload_32bit>& radix_sort::gpu::radix_pipeline<float, payload_32bit>::instance();

template radix_sort::gpu::radix_pipeline<int>::tmp_memory_info_t radix_sort::gpu::radix_pipeline<int>::calc_tmp_memory_info(const sort_info& info) const;
template radix_sort::gpu::radix_pipeline<uint32_t>::tmp_memory_info_t radix_sort::gpu::radix_pipeline<uint32_t>::calc_tmp_memory_info(const sort_info& info) const;
template radix_sort::gpu::radix_pipeline<float>::tmp_memory_info_t radix_sort::gpu::radix_pipeline<float>::calc_tmp_memory_info(const sort_info& info) const;
template radix_sort::gpu::radix_pipeline<float, payload_32bit>::tmp_memory_info_t radix_sort::gpu::radix_pipeline<float, payload_32bit>::calc_tmp_memory_info(const sort_info& info) const;

template void radix_sort::gpu::radix_pipeline<int>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const;
template void radix_sort::gpu::radix_pipeline<uint32_t>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const;
template void radix_sort::gpu::radix_pipeline<float>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const;
template void radix_sort::gpu::radix_pipeline<float, payload_32bit>::record_sort(VkCommandBuffer command_buffer, const sort_info& info) const;

template void radix_sort::gpu::radix_pipeline<int>::sort(const sort_info& info);
template void radix_sort::gpu::radix_pipeline<uint32_t>::sort(const sort_info& info);
template void radix_sort::gpu::radix_pipeline<float>::sort(const sort_info& info);
template void radix_sort::gpu::radix_pipeline<float, payload_32bit>::sort(const sort_info& info);

template void radix_sort::gpu::radix_pipeline<int>::sort(const sort_info_cpu& info);
template void radix_sort::gpu::radix_pipeline<uint32_t>::sort(const sort_info_cpu& info);
template void radix_sort::gpu::radix_pipeline<float>::sort(const sort_info_cpu& info);
template void radix_sort::gpu::radix_pipeline<float, payload_32bit>::sort(const sort_info_cpu& info);
