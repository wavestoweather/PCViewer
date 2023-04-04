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

#include "FFX_ParallelSort.h"
#include "radix_pipeline.hpp"
#include <shader_compiler.hpp>
#include <vk_util.hpp>

#include <numeric>
#include <random>
#include <vector>

//////////////////////////////////////////////////////////////////////////
// Helper functions for Vulkan

// Transition barrier
VkBufferMemoryBarrier buffer_transition(VkBuffer buffer, VkAccessFlags before, VkAccessFlags after, uint32_t size)
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

// Compile specified radix sort shader and create pipeline
VkPipeline compile_radix_pipeline(const std::string& shader_file, const robin_hood::unordered_map<std::string, std::string>& defines, std::string_view entry_point, const VkPipelineLayout pipeline_layout)
{
    std::string CompileFlags("-T cs_6_0");
#ifdef _DEBUG
    CompileFlags += " -Zi -Od";
#endif // _DEBUG

    auto spir_v = util::shader_compiler::compile(shader_file, defines);
    auto shader_module = util::vk::create_scoped_shader_module(spir_v);

    auto shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module, {}, {}, entry_point);
    auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(pipeline_layout, shader_stage_info);
    return util::vk::create_compute_pipeline(pipeline_info);
}


// Create all of the sort data for the sample
radix_sort::gpu::radix_pipeline::radix_pipeline()
{
    _MaxNumThreadgroups = 800;

    // Create Pipeline layout for Sorting
    // Only single layout needed, as all only use the push constants
    {
        auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(push_constants), 0);
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant_range);
        _SortPipelineLayout = util::vk::create_pipeline_layout(layout_info);
    }
        
    //////////////////////////////////////////////////////////////////////////
    // Create pipelines for radix sort
    {
        // SetupIndirectParams (indirect only)
        robin_hood::unordered_map<std::string, std::string> defines{{"VK_Const", "1"}};
        _FPSIndirectSetupParametersPipeline = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_SetupIndirectParameters", _SortPipelineLayout);

        _FPSCountPipeline       = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_Count", _SortPipelineLayout);
        _FPSCountReducePipeline = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_CountReduce", _SortPipelineLayout);
        _FPSScanPipeline        = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_Scan", _SortPipelineLayout);
        _FPSScanAddPipeline     = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_ScanAdd", _SortPipelineLayout);
        _FPSScatterPipeline     = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_Scatter", _SortPipelineLayout);
        
        // Radix scatter with payload (key and payload redistribution)
        defines["kRS_ValueCopy"] = "1";
        _FPSScatterPayloadPipeline = compile_radix_pipeline("ParallelSortCS.hlsl", defines, "FPS_Scatter", _SortPipelineLayout);
    }
}

// Perform Parallel Sort (radix-based sort)
void FFXParallelSort::Sort(VkCommandBuffer commandList, bool isBenchmarking, float benchmarkTime)
{
    bool bIndirectDispatch = m_UIIndirectSort;

    // To control which descriptor set to use for updating data
    static uint32_t frameCount = 0;
    uint32_t frameConstants = (++frameCount) % 3;

    std::string markerText = "FFXParallelSort";
    if (bIndirectDispatch) markerText += " Indirect";
    SetPerfMarkerBegin(commandList, markerText.c_str());

    // Buffers to ping-pong between when writing out sorted values
    VkBuffer* ReadBufferInfo(&m_DstKeyBuffers[0]), * WriteBufferInfo(&m_DstKeyBuffers[1]);
    VkBuffer* ReadPayloadBufferInfo(&m_DstPayloadBuffers[0]), * WritePayloadBufferInfo(&m_DstPayloadBuffers[1]);
    bool bHasPayload = m_UISortPayload;

    // Setup barriers for the run
    VkBufferMemoryBarrier Barriers[3];
    FFX_ParallelSortCB  constantBufferData = { 0 };

    // Fill in the constant buffer data structure (this will be done by a shader in the indirect version)
    uint32_t NumThreadgroupsToRun;
    uint32_t NumReducedThreadgroupsToRun;
    if (!bIndirectDispatch)
    {
        uint32_t NumberOfKeys = NumKeys[m_UIResolutionSize];
        FFX_ParallelSort_SetConstantAndDispatchData(NumberOfKeys, m_MaxNumThreadgroups, constantBufferData, NumThreadgroupsToRun, NumReducedThreadgroupsToRun);
    }
    else
    {
        struct SetupIndirectCB
        {
            uint32_t NumKeysIndex;
            uint32_t MaxThreadGroups;
        };
        SetupIndirectCB IndirectSetupCB;
        IndirectSetupCB.NumKeysIndex = m_UIResolutionSize;
        IndirectSetupCB.MaxThreadGroups = m_MaxNumThreadgroups;
            
        // Copy the data into the constant buffer
        VkDescriptorBufferInfo constantBuffer = m_pConstantBufferRing->AllocConstantBuffer(sizeof(SetupIndirectCB), (void*)&IndirectSetupCB);
        BindConstantBuffer(constantBuffer, m_SortDescriptorSetConstantsIndirect[frameConstants]);
            
        // Dispatch
        vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 1, 1, &m_SortDescriptorSetConstantsIndirect[frameConstants], 0, nullptr);
        vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 5, 1, &m_SortDescriptorSetIndirect, 0, nullptr);
        vkCmdBindPipeline(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_FPSIndirectSetupParametersPipeline);
        vkCmdDispatch(commandList, 1, 1, 1);
            
        // When done, transition the args buffers to INDIRECT_ARGUMENT, and the constant buffer UAV to Constant buffer
        VkBufferMemoryBarrier barriers[5];
        barriers[0] = BufferTransition(m_IndirectCountScatterArgs, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(uint32_t) * 3);
        barriers[1] = BufferTransition(m_IndirectReduceScanArgs, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(uint32_t) * 3);
        barriers[2] = BufferTransition(m_IndirectConstantBuffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, sizeof(FFX_ParallelSortCB));
        barriers[3] = BufferTransition(m_IndirectCountScatterArgs, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, sizeof(uint32_t) * 3);
        barriers[4] = BufferTransition(m_IndirectReduceScanArgs, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, sizeof(uint32_t) * 3);
        vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 5, barriers, 0, nullptr);
    }

    // Bind the scratch descriptor sets
    vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 4, 1, &m_SortDescriptorSetScratch, 0, nullptr);

    // Copy the data into the constant buffer and bind
    if (bIndirectDispatch)
    {
        //constantBuffer = m_IndirectConstantBuffer.GetResource()->GetGPUVirtualAddress();
        VkDescriptorBufferInfo constantBuffer;
        constantBuffer.buffer = m_IndirectConstantBuffer;
        constantBuffer.offset = 0;
        constantBuffer.range = VK_WHOLE_SIZE;
        BindConstantBuffer(constantBuffer, m_SortDescriptorSetConstants[frameConstants]);
    }
    else
    {
        VkDescriptorBufferInfo constantBuffer = m_pConstantBufferRing->AllocConstantBuffer(sizeof(FFX_ParallelSortCB), (void*)&constantBufferData);
        BindConstantBuffer(constantBuffer, m_SortDescriptorSetConstants[frameConstants]);
    }
    // Bind constants
    vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 0, 1, &m_SortDescriptorSetConstants[frameConstants], 0, nullptr);
        
    // Perform Radix Sort (currently only support 32-bit key/payload sorting
    uint32_t inputSet = 0;
    for (uint32_t Shift = 0; Shift < 32u; Shift += FFX_PARALLELSORT_SORT_BITS_PER_PASS)
    {
        // Update the bit shift
        vkCmdPushConstants(commandList, m_SortPipelineLayout, VK_SHADER_STAGE_ALL, 0, 4, &Shift);

        // Bind input/output for this pass
        vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 2, 1, &m_SortDescriptorSetInputOutput[inputSet], 0, nullptr);

        // Sort Count
        {
            vkCmdBindPipeline(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_FPSCountPipeline);

            if (bIndirectDispatch)
                vkCmdDispatchIndirect(commandList, m_IndirectCountScatterArgs, 0);                  
            else
                vkCmdDispatch(commandList, NumThreadgroupsToRun, 1, 1);
        }

        // UAV barrier on the sum table
        Barriers[0] = BufferTransition(m_FPSScratchBuffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, m_ScratchBufferSize);
        vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, Barriers, 0, nullptr);
            
        // Sort Reduce
        {
            vkCmdBindPipeline(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_FPSCountReducePipeline);
                
            if (bIndirectDispatch)
                vkCmdDispatchIndirect(commandList, m_IndirectReduceScanArgs, 0);
            else
                vkCmdDispatch(commandList, NumReducedThreadgroupsToRun, 1, 1);
                    
            // UAV barrier on the reduced sum table
            Barriers[0] = BufferTransition(m_FPSReducedScratchBuffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, m_ReducedScratchBufferSize);
            vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, Barriers, 0, nullptr);
        }

        // Sort Scan
        {
            // First do scan prefix of reduced values
            vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 3, 1, &m_SortDescriptorSetScanSets[0], 0, nullptr);
            vkCmdBindPipeline(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_FPSScanPipeline);

            if (!bIndirectDispatch)
            {
                assert(NumReducedThreadgroupsToRun < FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE && "Need to account for bigger reduced histogram scan");
            }
            vkCmdDispatch(commandList, 1, 1, 1);

            // UAV barrier on the reduced sum table
            Barriers[0] = BufferTransition(m_FPSReducedScratchBuffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, m_ReducedScratchBufferSize);
            vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, Barriers, 0, nullptr);
                
            // Next do scan prefix on the histogram with partial sums that we just did
            vkCmdBindDescriptorSets(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 3, 1, &m_SortDescriptorSetScanSets[1], 0, nullptr);
                
            vkCmdBindPipeline(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, m_FPSScanAddPipeline);
            if (bIndirectDispatch)
                vkCmdDispatchIndirect(commandList, m_IndirectReduceScanArgs, 0);
            else
                vkCmdDispatch(commandList, NumReducedThreadgroupsToRun, 1, 1);
        }

        // UAV barrier on the sum table
        Barriers[0] = BufferTransition(m_FPSScratchBuffer, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, m_ScratchBufferSize);
        vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, Barriers, 0, nullptr);
            
        // Sort Scatter
        {
            vkCmdBindPipeline(commandList, VK_PIPELINE_BIND_POINT_COMPUTE, bHasPayload ? m_FPSScatterPayloadPipeline : m_FPSScatterPipeline);

            if (bIndirectDispatch)
                vkCmdDispatchIndirect(commandList, m_IndirectCountScatterArgs, 0);
            else
                vkCmdDispatch(commandList, NumThreadgroupsToRun, 1, 1);
        }
            
        // Finish doing everything and barrier for the next pass
        int numBarriers = 0;
        Barriers[numBarriers++] = BufferTransition(*WriteBufferInfo, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(uint32_t) * NumKeys[2]);
        if (bHasPayload)
            Barriers[numBarriers++] = BufferTransition(*WritePayloadBufferInfo, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(uint32_t) * NumKeys[2]);
        vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, numBarriers, Barriers, 0, nullptr);
            
        // Swap read/write sources
        std::swap(ReadBufferInfo, WriteBufferInfo);
        if (bHasPayload)
            std::swap(ReadPayloadBufferInfo, WritePayloadBufferInfo);
        inputSet = !inputSet;
    }

    // When we are all done, transition indirect buffers back to UAV for the next frame (if doing indirect dispatch)
    if (bIndirectDispatch)
    {
        VkBufferMemoryBarrier barriers[3];
        barriers[0] = BufferTransition(m_IndirectConstantBuffer, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(FFX_ParallelSortCB));
        barriers[1] = BufferTransition(m_IndirectCountScatterArgs, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(uint32_t) * 3);
        barriers[2] = BufferTransition(m_IndirectReduceScanArgs, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, sizeof(uint32_t) * 3);
        vkCmdPipelineBarrier(commandList, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 3, barriers, 0, nullptr);
    }

    // Close out the perf capture
    SetPerfMarkerEnd(commandList);
}
