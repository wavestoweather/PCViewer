#include "distance_calculator.hpp"
#include <vk_initializers.hpp>
#include <vk_util.hpp>

namespace pipelines{

distance_calculator::distance_calculator(){
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _calculator_fence = util::vk::create_fence(fence_info);
}

const distance_calculator::pipeline_data& distance_calculator::_get_or_create_pipeline(const pipeline_specs& pipeline_specs){
    if(!_pipelines.contains(pipeline_specs)){
        pipeline_data& pipe_data = _pipelines[pipeline_specs];

        VkPushConstantRange push_constant_ranges{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants)};
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant_ranges);
        pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_info);

        auto shader_module = util::vk::create_shader_module(_compute_shader_path);
        std::vector<VkSpecializationMapEntry> specialization_entries{util::vk::initializers::specializationMapEntry(0, 0, sizeof(uint32_t))};
        std::vector<uint32_t> data_type_specialization{static_cast<uint32_t>(pipeline_specs.data_type)};
        auto specialization_info = util::vk::initializers::specializationInfo(specialization_entries, util::memory_view(data_type_specialization));
        auto shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, shader_module, &specialization_info);
        auto compute_info = util::vk::initializers::computePipelineCreateInfo(pipe_data.pipeline_layout, shader_stage_info);
        pipe_data.pipeline = util::vk::create_compute_pipeline(compute_info);

        vkDestroyShaderModule(globals::vk_context.device, shader_module, globals::vk_context.allocation_callbacks);
    }

    return _pipelines[pipeline_specs];
}

distance_calculator& distance_calculator::instance(){
    static distance_calculator calculator;
    return calculator;
}

void distance_calculator::calculate(const distance_info& info){
    const auto& pipeline_data = _get_or_create_pipeline(pipeline_specs{info.data_type});
    push_constants pc{};
    pc.data_header_address = info.data_header_address;
    pc.priority_distance_address = info.distances_address;
    pc.index_buffer_address = info.index_buffer_address;
    pc.data_size = info.data_size;
    pc.priority_attribute = info.priority_attribute;
    pc.priority_center = info.priority_center;
    pc.priority_distance = info.priority_distance;

    auto res = vkWaitForFences(globals::vk_context.device, 1, &_calculator_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    vkResetFences(globals::vk_context.device, 1, &_calculator_fence);

    if(_command_buffer)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
    _command_buffer = util::vk::create_begin_command_buffer(_command_pool);

    vkCmdPushConstants(_command_buffer, pipeline_data.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    vkCmdDispatch(_command_buffer, (info.data_size + 255) / 256, 1, 1);

    std::scoped_lock lock(*globals::vk_context.compute_mutex);
    util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue, info.gpu_sync_info.wait_semaphores, info.gpu_sync_info.wait_masks, info.gpu_sync_info.signal_semaphores, _calculator_fence);
}

void distance_calculator::wait_for_fence(uint64_t timeout){
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_calculator_fence, VK_TRUE, timeout); 
    util::check_vk_result(res);
}
}