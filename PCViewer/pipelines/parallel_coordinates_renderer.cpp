#include "parallel_coordinates_renderer.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <vma_util.hpp>
#include <file_util.hpp>
#include <array>
#include <parallel_coordinates_workbench.hpp>
#include <array_struct.hpp>
#include <global_descriptor_set_util.hpp>

namespace pipelines
{
parallel_coordinates_renderer::parallel_coordinates_renderer() 
{
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _render_fence = util::vk::create_fence(fence_info);
}

void parallel_coordinates_renderer::_pre_render_commands(VkCommandBuffer commands, const output_specs& output_specs)
{
    const auto& pipe_data = _pipelines[output_specs];
    auto begin_info = util::vk::initializers::renderPassBeginInfo(pipe_data.render_pass, pipe_data.framebuffer, {0, 0, output_specs.width, output_specs.height}, VkClearValue{});
    vkCmdBeginRenderPass(commands, &begin_info, {});
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_data.pipeline);
    vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_data.pipeline_layout, 0, 1, &globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->descriptor_set, 0, {});
    VkViewport viewport{};
    viewport.width = output_specs.width;
    viewport.height = output_specs.height;
    viewport.maxDepth = 1;
    vkCmdSetViewport(commands, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = {output_specs.width, output_specs.height};
    vkCmdSetScissor(commands, 0, 1, &scissor);
}

void parallel_coordinates_renderer::_post_render_commands(VkCommandBuffer commands, const output_specs& output_specs, VkFence fence, util::memory_view<VkSemaphore> wait_semaphores, util::memory_view<VkSemaphore> signal_semaphores)
{
    vkCmdEndRenderPass(commands);
    std::vector<VkPipelineStageFlags> stage_flags(wait_semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
    std::scoped_lock lock(globals::vk_context.graphics_mutex);
    util::vk::end_commit_command_buffer(commands, globals::vk_context.graphics_queue, wait_semaphores, stage_flags, signal_semaphores, fence);
}

const parallel_coordinates_renderer::pipeline_data& parallel_coordinates_renderer::get_or_create_pipeline(const output_specs& output_specs){
    if(!_pipelines.contains(output_specs)){
        std::cout << "[info] parallel_coordinates_renderer::get_or_create_pipeline() creating new pipeline for output_specs " << util::memory_view<const uint32_t>(util::memory_view(output_specs)) << std::endl;

        if(_pipelines.size() > max_pipeline_count){
            auto [pipeline, time] = *std::min_element(_pipeline_last_use.begin(), _pipeline_last_use.end(), [](const auto& l, const auto& r){return l.second < r.second;});
            auto [key, val] = *std::find_if(_pipelines.begin(), _pipelines.end(), [&](const auto& e){return e.second.pipeline == pipeline;});
            util::vk::destroy_pipeline(val.pipeline);
            util::vk::destroy_pipeline_layout(val.pipeline_layout);
            util::vk::destroy_framebuffer(val.framebuffer);
            util::vk::destroy_render_pass(val.render_pass);
            if(val.multi_sample_image)
                util::vk::destroy_image(val.multi_sample_image);
            if(val.multi_sample_view)
                util::vk::destroy_image_view(val.multi_sample_view);
            _pipeline_last_use.erase(pipeline);
            _pipelines.erase(key);
        }

        pipeline_data& pipe_data = _pipelines[output_specs];
        // creating the rendering buffers  -------------------------------------------------------------------------------
        // output image after multisample reduction (already given by parallel coordinates workbench)

        // multisample image
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            auto image_info = util::vk::initializers::imageCreateInfo(output_specs.format, {output_specs.width, output_specs.height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR, VK_IMAGE_TYPE_2D, 1, 1, output_specs.sample_count);
            auto allocation_info = util::vma::initializers::allocationCreateInfo();
            std::tie(pipe_data.multi_sample_image, pipe_data.multi_sample_view) = util::vk::create_image_with_view(image_info, allocation_info);
        }

        // render pass
        std::vector<VkAttachmentDescription> attachments;
        VkAttachmentDescription attachment = util::vk::initializers::attachmentDescription(output_specs.format);
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(attachment);
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            attachment.samples = output_specs.sample_count;
            attachments.push_back(attachment);
        }
        std::vector<VkAttachmentReference> attachment_references;
        VkAttachmentReference attachment_reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        attachment_references.push_back(attachment_reference);
        util::memory_view<VkAttachmentReference> resolve_reference{};
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            attachment_reference.attachment = 1;
            attachment_references.push_back(attachment_reference);
            resolve_reference = util::memory_view(attachment_reference);
        }
        auto subpass_description = util::vk::initializers::subpassDescription(VK_PIPELINE_BIND_POINT_GRAPHICS, {}, attachment_references, resolve_reference);

        auto render_pass_info = util::vk::initializers::renderPassCreateInfo(attachments, subpass_description);
        pipe_data.render_pass = util::vk::create_render_pass(render_pass_info);

        // framebuffer
        std::vector<VkImageView> image_views{output_specs.plot_image_view};
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT)
            image_views.push_back(pipe_data.multi_sample_view);
        auto framebuffer_info = util::vk::initializers::framebufferCreateInfo(pipe_data.render_pass, image_views, output_specs.width, output_specs.height, 1);
        pipe_data.framebuffer = util::vk::create_framebuffer(framebuffer_info);
        // creating the rendering pipeline -------------------------------------------------------------------------------

        // pipeline layout creation
        auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants), 0);
        assert(globals::descriptor_sets.contains(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
        auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout, util::memory_view(push_constant_range));
        pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

        // pipeline creation
        auto pipeline_rasterizer = util::vk::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_LINE, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipeline_rasterizer.lineWidth = 1;

        auto pipeline_color_blend_attachment = util::vk::initializers::pipelineColorBlendAttachmentStateStandardAlphaBlend();

        auto pipeline_color_blend = util::vk::initializers::pipelineColorBlendStateCreateInfo(pipeline_color_blend_attachment);

        auto pipeline_viewport = util::vk::initializers::pipelineViewportStateCreateInfo(1, 1);

        std::vector<VkDynamicState> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        auto pipeline_dynamic_states = util::vk::initializers::pipelineDynamicStateCreateInfo(dynamic_states); 

        auto pipeline_depth_stencil = util::vk::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

        auto pipeline_multi_sample = util::vk::initializers::pipelineMultisampleStateCreateInfo(output_specs.sample_count);

        auto pipeline_vertex_state = util::vk::initializers::pipelineVertexInputStateCreateInfo();//(vertex_input_binding, vertex_input_attribute);

        switch(output_specs.render_typ){
        case structures::parallel_coordinates_renderer::render_type::polyline_spline:{
            VkShaderModule vertex_module = util::vk::create_shader_module(vertex_shader_path); 
            VkShaderModule fragment_module = util::vk::create_shader_module(fragment_shader_path);  

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module);
            shader_infos[1] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module);

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, 0, VK_FALSE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, pipe_data.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipline(pipeline_create_info);

            vkDestroyShaderModule(globals::vk_context.device, vertex_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);
            break;
        }
        case structures::parallel_coordinates_renderer::render_type::large_vis_lines:{
            VkShaderModule vertex_module = util::vk::create_shader_module(large_vis_vertex_shader_path); 
            VkShaderModule fragment_module = util::vk::create_shader_module(fragment_shader_path); 

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module);
            shader_infos[1] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module);

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY, 0, VK_TRUE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, pipe_data.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipline(pipeline_create_info);

            vkDestroyShaderModule(globals::vk_context.device, vertex_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);
            break;
        }
        default:
            throw std::runtime_error("parallel_coordinates_renderer::get_or_create_pipeline() render_type " + (std::string)structures::parallel_coordinates_renderer::render_type_names[output_specs.render_typ] + " not yet implemented.");
        }
    }
    _pipeline_last_use[_pipelines[output_specs].pipeline] = std::chrono::system_clock::now();
    return _pipelines[output_specs];
}

const structures::buffer_info& parallel_coordinates_renderer::get_or_resize_info_buffer(size_t byte_size){
    if(byte_size > _attribute_info_buffer_size){
        if(_attribute_info_buffer)
            util::vk::destroy_buffer(_attribute_info_buffer);
        
        auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, byte_size);
        auto allocation_info = util::vma::initializers::allocationCreateInfo(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        _attribute_info_buffer = util::vk::create_buffer(buffer_info, allocation_info);
    }
    return _attribute_info_buffer;
}

parallel_coordinates_renderer& parallel_coordinates_renderer::instance(){
    static parallel_coordinates_renderer renderer;
    return renderer;
}

void parallel_coordinates_renderer::render(const render_info& info){
    struct attribute_infos{
	    uint 	attribute_count;				// amount of active attributes
	    uint 	_, __;
	    uint 	data_flags;
    };

    std::vector<uint32_t> active_attribute_indices;     // these inlcude the order
    for(int place_index: util::size_range(info.workbench.attributes_order_info.read()))
        if(info.workbench.attributes_order_info.read()[place_index].active)
            active_attribute_indices.push_back(info.workbench.attributes_order_info.read()[place_index].attribut_index);

    const auto& drawlists = globals::drawlists.read();
    auto first_dl = info.workbench.drawlist_infos.read()[0].drawlist_id;
    auto data_type = globals::drawlists.read().at(first_dl).read().dataset_read().data_flags.half ? structures::parallel_coordinates_renderer::data_type::half_t: structures::parallel_coordinates_renderer::data_type::float_t;
    output_specs out_specs{
        info.workbench.plot_data.read().width, 
        info.workbench.plot_data.read().height, 
        info.workbench.plot_data.read().image_samples, 
        info.workbench.plot_data.read().image_format, 
        info.workbench.render_type.read(), 
        info.workbench.plot_data.read().image_view, 
        data_type}; 
    auto pipeline_info = get_or_create_pipeline(out_specs);

    structures::dynamic_struct<attribute_infos, ImVec4> attribute_infos(active_attribute_indices.size());
    attribute_infos->attribute_count = active_attribute_indices.size();
    attribute_infos->data_flags = {};
    for(int active_attribute_index: util::size_range(active_attribute_indices)){
        uint32_t cur_attribute_index = active_attribute_indices[active_attribute_index];
        attribute_infos[active_attribute_index].x = float(cur_attribute_index);
        attribute_infos[active_attribute_index].y = info.workbench.attributes.read()[cur_attribute_index].bounds.read().min;
        attribute_infos[active_attribute_index].z = info.workbench.attributes.read()[cur_attribute_index].bounds.read().max;
    }
    auto attribute_infos_gpu = get_or_resize_info_buffer(attribute_infos.data().byteSize());

    auto res = vkWaitForFences(globals::vk_context.device, 1, &_render_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);  // wait indefenitely for prev rendering
    vkResetFences(globals::vk_context.device, 1, &_render_fence);

    util::vma::upload_data(attribute_infos.data(), attribute_infos_gpu);
    if(_render_commands.size())
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, _render_commands.size(), _render_commands.data());
    _render_commands.resize(1);
    _render_commands[0] = util::vk::create_begin_command_buffer(_command_pool);
    _pre_render_commands(_render_commands[0], out_specs);
    VkClearAttachment clear_value{};
    clear_value.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear_value.clearValue.color.float32[3] = 1;
    VkClearRect clear_rect{};
    clear_rect.layerCount = 1;
    clear_rect.rect.extent = {out_specs.width, out_specs.height};
    vkCmdClearAttachments(_render_commands[0], 1, &clear_value, 1, &clear_rect);

    size_t batch_size{};
    switch(info.workbench.render_strategy){
    case workbenches::parallel_coordinates_workbench::render_strategy::all:
        for(const auto& dl: info.workbench.drawlist_infos.read()){
            const auto& ds = globals::drawlists.read().at(dl.drawlist_id).read().dataset_read();
            batch_size += ds.data_size;
        }
        break;
    case workbenches::parallel_coordinates_workbench::render_strategy::batched:
        batch_size = info.workbench.render_batch_size;
        break;
    }

    size_t cur_batch_lines{};
    for(const auto& dl: info.workbench.drawlist_infos.read()){
        const auto& ds = drawlists.at(dl.drawlist_id).read().dataset_read();

        size_t data_size = globals::drawlists.read().at(dl.drawlist_id).read().const_templatelist().indices.size();
        size_t cur_batch_size = std::min(data_size, batch_size);
        size_t cur_offset = 0;
        do{
            push_constants pc{};
            pc.attribute_info_address = util::vk::get_buffer_address(_attribute_info_buffer);
            pc.data_header_address = util::vk::get_buffer_address(ds.gpu_data.header);
            pc.index_buffer_address = util::vk::get_buffer_address(dl.drawlist_read().const_templatelist().gpu_indices);
            pc.identity_index = uint(dl.drawlist_read().const_templatelist().flags.identity_indices);
            pc.vertex_count_per_line = active_attribute_indices.size();
            pc.color = dl.drawlist_read().appearance_drawlist.read().color;
            vkCmdPushConstants(_render_commands.back(), pipeline_info.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdDraw(_render_commands.back(), pc.vertex_count_per_line, cur_batch_size, 0, cur_offset);

            cur_offset += cur_batch_size;
            cur_batch_lines += cur_batch_size;
            if(cur_batch_lines >= batch_size){
                // dispatching command buffer
                _post_render_commands(_render_commands.back(), out_specs);
                _render_commands.push_back(util::vk::create_begin_command_buffer(_command_pool));
                _pre_render_commands(_render_commands.back(), out_specs);
                cur_batch_lines = 0;
            }
        } while(cur_offset < data_size);
    }

    // committing last command buffer
    _post_render_commands(_render_commands.back(), out_specs, _render_fence);    
}
}