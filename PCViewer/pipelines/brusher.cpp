#include "brusher.hpp"

#include <vk_util.hpp>
#include <vma_util.hpp>
#include <vk_initializers.hpp>
#include <file_util.hpp>
#include <drawlists.hpp>

namespace pipelines
{
brusher::brusher() 
{
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _brush_fence = util::vk::create_fence(fence_info);

    // pipeline creation
    auto shader_module = util::vk::create_shader_module(compute_shader_path);
    auto stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, shader_module);

    VkPushConstantRange push_constant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants)};
    auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
    _brushing_pipeline_layout = util::vk::create_pipeline_layout(layout_info);

    auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(_brushing_pipeline_layout, stage_create_info);
    _brushing_pipeline = util::vk::create_compute_pipline(pipeline_info);

    vkDestroyShaderModule(globals::vk_context.device, shader_module, globals::vk_context.allocation_callbacks);
}

brusher& brusher::instance() 
{
    static brusher singleton;
    return singleton;
}

void brusher::brush(const brush_info& info) 
{
    const auto& dl = globals::drawlists.read().at(info.drawlist_id);
    const auto& ds = dl.read().dataset_read();
    const auto& tl = dl.read().const_templatelist();
    const bool no_brushes = dl.read().local_brushes.read().empty() && globals::global_brushes.read().empty();
    push_constants pc{};
    pc.local_brush_address = util::vk::get_buffer_address(dl.read().local_brushes.read().brushes_gpu);
    pc.global_brush_address = dl.read().immune_to_global_brushes.read() ? VkDeviceAddress{}: util::vk::get_buffer_address(globals::global_brushes.read().brushes_gpu);
    pc.active_indices_address = util::vk::get_buffer_address(dl.read().active_indices_bitset_gpu);
    pc.data_address = util::vk::get_buffer_address(ds.gpu_data.header);
    pc.index_buffer_address = util::vk::get_buffer_address(tl.gpu_indices);
    pc.local_global_brush_combine = static_cast<uint32_t>(info.brush_comb);
    pc.data_size = tl.data_size;

    auto res = vkWaitForFences(globals::vk_context.device, 1, &_brush_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    res = vkResetFences(globals::vk_context.device, 1, &_brush_fence); util::check_vk_result(res);
    if(_command_buffer)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
    _command_buffer = util::vk::create_begin_command_buffer(_command_pool);
    vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, _brushing_pipeline);
    vkCmdPushConstants(_command_buffer, _brushing_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t dispatch_x = ((pc.data_size + 31) / 32 + shader_local_size - 1) / shader_local_size;
    if(no_brushes)
        vkCmdFillBuffer(_command_buffer, dl.read().active_indices_bitset_gpu.buffer, 0, VK_WHOLE_SIZE, uint32_t(-1));
    else
        vkCmdDispatch(_command_buffer, dispatch_x, 1, 1);
    util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue, info.wait_semaphores, info.wait_flags, info.signal_semaphores, _brush_fence);
    res = vkWaitForFences(globals::vk_context.device, 1, &_brush_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);

}
}