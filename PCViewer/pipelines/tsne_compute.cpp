#include "tsne_compute.hpp"
#include <vk_initializers.hpp>
#include <vk_util.hpp>
#include <util.hpp>

namespace pipelines{
tsne_compute::tsne_compute(){
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _fence = util::vk::create_fence(fence_info);
}

const tsne_compute::pipeline_data& tsne_compute::_get_or_create_pipeline(){
    if(!_pipeline_data.pipeline){
        auto shader_module = util::vk::create_scoped_shader_module(_compute_shader_path);
        auto stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        
        VkPushConstantRange push_constant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants)};
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.layout = util::vk::create_pipeline_layout(layout_info);

        auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.layout, stage_create_info);
        _pipeline_data.pipeline = util::vk::create_compute_pipeline(pipeline_info);
    }
    return _pipeline_data;
}

tsne_compute& tsne_compute::instance(){
    static tsne_compute singleton;
    return singleton;
}

void tsne_compute::record(VkCommandBuffer commands, const tsne_compute_info& info){
    const uint32_t dispatch_x = util::align(info.datapoint_count, _workgroup_size);
    const auto& pipeline_data = _get_or_create_pipeline();
    vkCmdPushConstants(commands, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(info), &info);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdDispatch(commands, dispatch_x, 1, 1);
    // TODO missing pipelines
}

void tsne_compute::calculate(const tsne_compute_info& info){
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    res = vkResetFences(globals::vk_context.device, 1, &_fence); util::check_vk_result(res);
    if(_command_buffer)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
    _command_buffer = util::vk::create_begin_command_buffer(_command_pool);
    record(_command_buffer, info);
    util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue.const_access().get(), {}, {}, {}, _fence);
}

void tsne_compute::wait_for_fence(uint64_t timeout){
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_fence, VK_TRUE, timeout);
    util::check_vk_result(res);
}
}