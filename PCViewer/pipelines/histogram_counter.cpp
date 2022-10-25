#include "histogram_counter.hpp"
#include <vk_util.hpp>
#include <vk_initializers.hpp>

namespace pipelines{

histogram_counter::histogram_counter(){
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _count_fence = util::vk::create_fence(fence_info);
}

const histogram_counter::pipeline_data& histogram_counter::_get_or_create_pipeline(const pipeline_specs& pipeline_specs){
    if(!_pipelines.contains(pipeline_specs)){

        pipeline_data& pipe_data = _pipelines[pipeline_specs];

        VkPushConstantRange push_constants{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push_constants};
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constants);
        pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_info);

        auto shader_module = util::vk::create_shader_module(_compute_shader_path);
        std::vector<VkSpecializationMapEntry> specialization_entries{util::vk::initializers::specializationMapEntry(0, 0, sizeof(uint32_t))};
        uint32_t data_type_specialization{static_cast<uint32_t>(pipeline_specs.d_type)};
        auto specialization_info = util::vk::initializers::specializationInfo(specialization_entries, util::memory_view(data_type_specialization));
        auto shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, shader_module, &specialization_info);
        auto compute_info = util::vk::initializers::computePipelineCreateInfo(pipe_data.pipeline_layout, shader_stage_info);

        vkDestroyShaderModule(globals::vk_context.device, shader_module, globals::vk_context.allocation_callbacks);
    }

    return _pipelines[pipeline_specs];
}

histogram_counter& histogram_counter::instance(){
    static histogram_counter counter;
    return counter;
}

void histogram_counter::count(const count_info& info){
    const auto& pipeline_data = _get_or_create_pipeline(pipeline_specs{info.data_type});
    push_constants pc{};
    pc.data_header_address = info.data_header_address;
    pc.gpu_data_activations = info.gpu_data_activations;
    pc.a1 = info.column_indices[0];
    pc.a2 = info.column_indices[1];
    if(info.column_indices.size() > 2)
        pc.a3 = info.column_indices[2];
    if(info.column_indices.size() > 3)
        pc.a4 = info.column_indices[3];
    pc.s1 = info.bin_sizes[0];
    pc.s2 = info.bin_sizes[1];
    if(info.bin_sizes.size() > 2)
        pc.s3 = info.bin_sizes[2];
    if(info.bin_sizes.size() > 3)
        pc.s4 = info.bin_sizes[3];
    pc.data_size = info.data_size;

    auto res = vkWaitForFences(globals::vk_context.device, 1, &_count_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    vkResetFences(globals::vk_context.device, 1, &_count_fence);

    if(_command_buffer)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
    _command_buffer = util::vk::create_begin_command_buffer(_command_pool);

    vkCmdPushConstants(_command_buffer, pipeline_data.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(_command_buffer, (info.data_size + 255) / 256, 1, 1);

    std::scoped_lock lock(*globals::vk_context.compute_mutex);
    util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue, info.gpu_sync_info.wait_semaphores, info.gpu_sync_info.wait_masks, info.gpu_sync_info.signal_semaphores, _count_fence);
}

}