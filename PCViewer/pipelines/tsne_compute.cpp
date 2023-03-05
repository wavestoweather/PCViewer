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
    if(!_pipeline_data.charges_qij_pipeline){
        auto shader_module = util::vk::create_scoped_shader_module(_charges_qij_path);
        auto stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        VkPushConstantRange push_constant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_charges_qij)};
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.charges_qij_layout = util::vk::create_pipeline_layout(layout_info);
        auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.charges_qij_layout, stage_create_info);
        _pipeline_data.charges_qij_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_copy_from_fft_output_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_copy_fft)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.copy_fft_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.copy_fft_layout, stage_create_info);
        _pipeline_data.copy_fft_from_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_copy_to_fft_input_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.copy_fft_layout, stage_create_info);
        _pipeline_data.copy_fft_to_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_copy_to_w_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_copy_to_w)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.copy_to_w_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.copy_to_w_layout, stage_create_info);
        _pipeline_data.copy_to_w_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_integration_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_integration)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.integration_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.integration_layout, stage_create_info);
        _pipeline_data.integration_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_interpolate_device_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_interpolate)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.interpolate_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.interpolate_layout, stage_create_info);
        _pipeline_data.interpolate_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_interpolate_indices_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_interpolate_indices)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.interpolate_indices_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.interpolate_indices_layout, stage_create_info);
        _pipeline_data.interpolate_indices_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_kernel_tilde_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_kernel_tilde)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.kernel_tilde_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.kernel_tilde_layout, stage_create_info);
        _pipeline_data.kernel_tilde_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_perplexity_search_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_perplexity_search)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.perplexity_search_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.perplexity_search_layout, stage_create_info);
        _pipeline_data.perplexity_search_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_qij_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij_qij)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_qij_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_qij_layout, stage_create_info);
        _pipeline_data.pij_qij_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_qij2_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij_qij2)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_qij2_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_qij2_layout, stage_create_info);
        _pipeline_data.pij_qij2_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_qij3_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij_qij3)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_qij3_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_qij3_layout, stage_create_info);
        _pipeline_data.pij_qij3_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_layout, stage_create_info);
        _pipeline_data.pij_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_point_box_index_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_point_box_index)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.point_box_index_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.point_box_index_layout, stage_create_info);
        _pipeline_data.point_box_index_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_potential_indices_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_potential_indices)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.potential_indices_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.potential_indices_layout, stage_create_info);
        _pipeline_data.potential_indices_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_repulsive_forces_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_repulsive_forces)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.repulsive_forces_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.repulsive_forces_layout, stage_create_info);
        _pipeline_data.repulsive_forces_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_sum_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_sum)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.sum_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.sum_layout, stage_create_info);
        _pipeline_data.sum_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_upper_lower_bounds_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_upper_lower_bounds)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.upper_lower_bounds_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.upper_lower_bounds_layout, stage_create_info);
        _pipeline_data.upper_lower_bounds_pipeline = util::vk::create_compute_pipeline(pipeline_info);
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
    //vkCmdPushConstants(commands, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(info), &info);
    //vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    //vkCmdDispatch(commands, dispatch_x, 1, 1);
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