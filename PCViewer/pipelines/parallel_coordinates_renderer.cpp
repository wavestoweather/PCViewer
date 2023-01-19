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
#include <histogram_registry_util.hpp>
#include <splines.hpp>
#include <priority_globals.hpp>
#include <data_util.hpp>

namespace pipelines
{
parallel_coordinates_renderer::parallel_coordinates_renderer() 
{
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _render_fence = util::vk::create_fence(fence_info);
}

void parallel_coordinates_renderer::_pre_render_commands(VkCommandBuffer commands, const output_specs& output_specs, bool clear_framebuffer, const ImVec4& clear_color)
{
    const auto& pipe_data = _pipelines[output_specs];
    auto begin_info = util::vk::initializers::renderPassBeginInfo(pipe_data.render_pass, pipe_data.framebuffer, {0, 0, output_specs.width, output_specs.height});
    vkCmdBeginRenderPass(commands, &begin_info, {});
    
    if(clear_framebuffer){
        VkClearAttachment clear_value{};
        clear_value.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clear_value.clearValue.color.float32[0] = clear_color.x;
        clear_value.clearValue.color.float32[1] = clear_color.y;
        clear_value.clearValue.color.float32[2] = clear_color.z;
        clear_value.clearValue.color.float32[3] = clear_color.w;
        VkClearRect clear_rect{};
        clear_rect.layerCount = 1;
        clear_rect.rect.extent = {output_specs.width, output_specs.height};
        vkCmdClearAttachments(_render_commands[0], 1, &clear_value, 1, &clear_rect);
    }
    
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

void parallel_coordinates_renderer::_post_render_commands(VkCommandBuffer commands, VkFence fence, util::memory_view<VkSemaphore> wait_semaphores, util::memory_view<VkSemaphore> signal_semaphores)
{
    std::vector<VkPipelineStageFlags> stage_flags(wait_semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
    std::scoped_lock lock(*globals::vk_context.graphics_mutex);
    util::vk::end_commit_command_buffer(commands, globals::vk_context.graphics_queue, wait_semaphores, stage_flags, signal_semaphores, fence);
}

const parallel_coordinates_renderer::pipeline_data& parallel_coordinates_renderer::get_or_create_pipeline(const output_specs& output_specs){
    if(!_pipelines.contains(output_specs)){
        if(logger.logging_level >= logging::level::l_4){
            std::stringstream ss; ss << util::memory_view<const uint32_t>(util::memory_view(output_specs));
            logger << "[info] parallel_coordinates_renderer::get_or_create_pipeline() creating new pipeline for output_specs " << ss.str() << logging::endl;
        }

        multisample_key ms_key{output_specs.format, output_specs.sample_count, output_specs.width, output_specs.height};
        if(_pipelines.size() > max_pipeline_count){
            auto [pipeline, time] = *std::min_element(_pipeline_last_use.begin(), _pipeline_last_use.end(), [](const auto& l, const auto& r){return l.second < r.second;});
            VkPipeline pipe = pipeline; // needed for msvc and clang on windows as structured bindings can not be inserted to lambda...
            auto [key, val] = *std::find_if(_pipelines.begin(), _pipelines.end(), [&](const auto& e){return e.second.pipeline == pipe;});
            util::vk::destroy_pipeline(val.pipeline);
            util::vk::destroy_pipeline_layout(val.pipeline_layout);
            util::vk::destroy_framebuffer(val.framebuffer);
            util::vk::destroy_render_pass(val.render_pass);
            if(--_multisample_images[ms_key].count == 0){
                util::vk::destroy_image(_multisample_images[ms_key].image);
                util::vk::destroy_image_view(_multisample_images[ms_key].image_view);
            }
            _pipeline_last_use.erase(pipeline);
            _pipelines.erase(key);
        }

        pipeline_data& pipe_data = _pipelines[output_specs];
        // creating the rendering buffers  -------------------------------------------------------------------------------
        // output image after multisample reduction (already given by parallel coordinates workbench)

        // multisample image
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            if(!_multisample_images.contains(ms_key)){
                auto image_info = util::vk::initializers::imageCreateInfo(output_specs.format, {output_specs.width, output_specs.height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TYPE_2D, 1, 1, output_specs.sample_count);
                auto allocation_info = util::vma::initializers::allocationCreateInfo();
                std::tie(_multisample_images[ms_key].image, _multisample_images[ms_key].image_view) = util::vk::create_image_with_view(image_info, allocation_info);

                // updating the image layout
                auto image_barrier = util::vk::initializers::imageMemoryBarrier(_multisample_images[ms_key].image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                util::vk::convert_image_layouts_execute(image_barrier);
            }
            _multisample_images[ms_key].count++;
        }

        // render pass
        std::vector<VkAttachmentDescription> attachments;
        VkAttachmentDescription attachment = util::vk::initializers::attachmentDescription(output_specs.format);
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments.push_back(attachment);
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachment.samples = output_specs.sample_count;
            attachments.push_back(attachment);
        }
        std::vector<VkAttachmentReference> attachment_references;
        VkAttachmentReference attachment_reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        attachment_references.push_back(attachment_reference);
        util::memory_view<VkAttachmentReference> resolve_reference{};
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            attachment_references.back().attachment = 1;    // normal rendering goes to attachment 1 in multisampling case (at index 1 multisample image is attached)
            attachment_reference.attachment = 0;
            resolve_reference = util::memory_view(attachment_reference);
        }
        auto subpass_description = util::vk::initializers::subpassDescription(VK_PIPELINE_BIND_POINT_GRAPHICS, {}, attachment_references, resolve_reference);

        auto render_pass_info = util::vk::initializers::renderPassCreateInfo(attachments, subpass_description);
        pipe_data.render_pass = util::vk::create_render_pass(render_pass_info);

        // framebuffer
        std::vector<VkImageView> image_views{output_specs.plot_image_view};
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT)
            image_views.push_back(_multisample_images[ms_key].image_view);
        auto framebuffer_info = util::vk::initializers::framebufferCreateInfo(pipe_data.render_pass, image_views, output_specs.width, output_specs.height, 1);
        pipe_data.framebuffer = util::vk::create_framebuffer(framebuffer_info);
        // creating the rendering pipeline -------------------------------------------------------------------------------

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

        auto specialization_map_entry = util::vk::initializers::specializationMapEntry(0, 0, sizeof(output_specs.data_typ));
        auto specialization_info = util::vk::initializers::specializationInfo(specialization_map_entry, util::memory_view(output_specs.data_typ));

        switch(output_specs.render_typ){
        case structures::parallel_coordinates_renderer::render_type::polyline_spline:{
            // pipeline layout creation
            auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants), 0);
            assert(globals::descriptor_sets.count(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout, util::memory_view(push_constant_range));
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            VkShaderModule vertex_module = util::vk::create_shader_module(vertex_shader_path); 
            VkShaderModule fragment_module = util::vk::create_shader_module(fragment_shader_path);  

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module, &specialization_info);
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

            pipe_data.pipeline = util::vk::create_graphics_pipeline(pipeline_create_info);

            vkDestroyShaderModule(globals::vk_context.device, vertex_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);
            break;
        }
        case structures::parallel_coordinates_renderer::render_type::large_vis_lines:{
            // pipeline layout creation
            auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants_large_vis), 0);
            assert(globals::descriptor_sets.count(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout, util::memory_view(push_constant_range));
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            VkShaderModule vertex_module = util::vk::create_shader_module(large_vis_vertex_shader_path); 
            VkShaderModule fragment_module = util::vk::create_shader_module(fragment_shader_path); 

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module, &specialization_info),
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module)
            };

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 0, VK_FALSE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, pipe_data.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipeline(pipeline_create_info);

            vkDestroyShaderModule(globals::vk_context.device, vertex_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);
            break;
        }
        case structures::parallel_coordinates_renderer::render_type::histogram:{
            std::array<VkPushConstantRange, 2> push_constant_ranges{
                util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants_hist_vert), 0),                                 // vertex shader push constants
                util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(push_constants_hist_frag), sizeof(push_constants_hist_vert)) // fragment shader push constants
            };
            assert(globals::descriptor_sets.count(util::global_descriptors::heatmap_descriptor_id));
            auto heatmap_layout = globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout;
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(heatmap_layout, push_constant_ranges);
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            auto vertex_module = util::vk::create_scoped_shader_module(histogram_vertex_shader_path);
            auto fragment_module = util::vk::create_scoped_shader_module(histogram_fragment_shader_path);

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, *vertex_module, &specialization_info),
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, *fragment_module)
            };

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, 0, VK_FALSE);

            auto pipeline_fill_rasterizer = util::vk::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, pipe_data.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_fill_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipeline(pipeline_create_info);
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
    if(byte_size > _attribute_info_buffer.size){
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
    struct attribute_infos_t{
        uint32_t     attribute_count;                // amount of active attributes
        uint32_t     _t, __t;
        uint32_t     data_flags;
    };

    const auto active_ordered_attributes = info.workbench.get_active_ordered_attributes();     // these inlcude the order
    //std::vector<std::string_view> active_attribute_indices = info.workbench.get_active_ordered_indices();     // these inlcude the order

    const auto& drawlists = globals::drawlists.read();

    std::map<std::string_view, size_t> ds_attribute_info_offsets;
    std::vector<uint8_t> info_bytes;
    size_t infos_byte_size{};
    for(const auto& dl_ref: info.workbench.drawlist_infos.read()){
        const auto& ds = dl_ref.dataset_read();
        const auto active_attribute_indices = util::data::active_attribute_refs_to_indices(active_ordered_attributes, ds.attributes);
        structures::dynamic_struct<attribute_infos_t, ImVec4> attribute_info(active_ordered_attributes.size());
        attribute_info->attribute_count = active_ordered_attributes.size();
        attribute_info->data_flags = {};
        for(int active_attribute_index: util::size_range(active_attribute_indices)){
            uint32_t cur_attribute_index = active_attribute_indices[active_attribute_index];
            attribute_info[active_attribute_index].x = float(cur_attribute_index);
            attribute_info[active_attribute_index].y = active_ordered_attributes[active_attribute_index].get().bounds->read().min;
            attribute_info[active_attribute_index].z = active_ordered_attributes[active_attribute_index].get().bounds->read().max;
        }
        ds_attribute_info_offsets[ds.id] = infos_byte_size;
        infos_byte_size += attribute_info.byte_size();
        info_bytes.insert(info_bytes.end(), attribute_info.data().begin(), attribute_info.data().end());
    }

    auto res = vkWaitForFences(globals::vk_context.device, 1, &_render_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);  // wait indefenitely for prev rendering
    vkResetFences(globals::vk_context.device, 1, &_render_fence);

    auto attribute_infos_gpu = get_or_resize_info_buffer(infos_byte_size);

    util::vma::upload_data(info_bytes, attribute_infos_gpu);
    if(_render_commands.size())
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, _render_commands.size(), _render_commands.data());
    _render_commands.resize(1);
    _render_commands[0] = util::vk::create_begin_command_buffer(_command_pool);

    size_t batch_size{};
    switch(info.workbench.render_strategy){
    case workbenches::parallel_coordinates_workbench::render_strategy::all:
        for(const auto& dl: info.workbench.drawlist_infos.read()){
            const auto& ds = dl.drawlist_read().dataset_read();
            batch_size += ds.data_size;
        }
        break;
    case workbenches::parallel_coordinates_workbench::render_strategy::batched:
        batch_size = info.workbench.setting.read().render_batch_size;
        break;
    }

    size_t cur_batch_lines{};
    bool clear_framebuffer = true;
    for(const auto& dl: info.workbench.drawlist_infos.read()){
        if(!dl.appearance->read().show)
            continue;
        const auto& drawlist = dl.drawlist_read();
        const auto& ds = drawlist.dataset_read();

        output_specs out_specs{
            info.workbench.plot_data.read().image_view,
            info.workbench.plot_data.read().image_format, 
            info.workbench.plot_data.read().image_samples, 
            info.workbench.plot_data.read().width, 
            info.workbench.plot_data.read().height, 
            drawlist.const_templatelist().data_size < info.workbench.setting.read().histogram_rendering_threshold ? structures::parallel_coordinates_renderer::render_type::polyline_spline: structures::parallel_coordinates_renderer::render_type::large_vis_lines, 
            ds.data_flags.data_type
        };
        auto pipeline_info = get_or_create_pipeline(out_specs);

        _pre_render_commands(_render_commands.back(), out_specs, clear_framebuffer, info.workbench.setting.read().plot_background);
        clear_framebuffer = false;

        size_t data_size = drawlist.const_templatelist().data_size;
        if(out_specs.render_typ == structures::parallel_coordinates_renderer::render_type::large_vis_lines)
            data_size = out_specs.height;
        size_t cur_batch_size = std::min(data_size, batch_size);
        size_t cur_offset = 0;
        do{
            switch(out_specs.render_typ){
            case structures::parallel_coordinates_renderer::render_type::polyline_spline:{
                push_constants pc{};
                pc.attribute_info_address = util::vk::get_buffer_address(attribute_infos_gpu) + ds_attribute_info_offsets[ds.id];
                pc.data_header_address = util::vk::get_buffer_address(ds.gpu_data.header);
                if(dl.priority_render.read())
                    pc.priorities_address = util::vk::get_buffer_address(drawlist.priority_colors_gpu);
                pc.index_buffer_address = util::vk::get_buffer_address(drawlist.const_templatelist().gpu_indices);
                if(dl.priority_render.read())
                    pc.index_order_address = util::vk::get_buffer_address(drawlist.priority_indices.at(std::string(globals::priority_drawlist_standard_order)));
                pc.activation_bitset_address = util::vk::get_buffer_address(drawlist.active_indices_bitset_gpu);
                pc.vertex_count_per_line = (active_ordered_attributes.size() - 1) * (info.workbench.setting.read().render_splines ? _spline_resolution: 1) + 1;
                pc.color = dl.appearance->read().color;
                vkCmdPushConstants(_render_commands.back(), pipeline_info.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                vkCmdDraw(_render_commands.back(), pc.vertex_count_per_line, cur_batch_size, 0, cur_offset);
                break;
            }
            case structures::parallel_coordinates_renderer::render_type::large_vis_lines:{
                auto indices = util::data::active_attribute_refs_to_indices(active_ordered_attributes, ds.attributes);
                int height = info.workbench.plot_data.read().height;
                std::vector<int> bin_sizes = info.workbench.setting.read().render_splines ? std::vector<int>{config::histogram_splines_hidden_res, height, height, config::histogram_splines_hidden_res}: std::vector<int>{height, height};
                size_t lines_amt = 1;
                for(uint32_t s: bin_sizes)
                    lines_amt *= s;

                for(size_t i: util::i_range(indices.size() - 1)){
                    std::vector<uint32_t> hist_indices;
                    std::vector<std::string_view> attribute_names;
                    std::vector<structures::min_max<float>> attribute_bounds; 

                    push_constants_large_vis pc{};
                    pc.attribute_info_address = util::vk::get_buffer_address(attribute_infos_gpu) + ds_attribute_info_offsets[ds.id];
                    if(info.workbench.setting.read().render_splines){
                        hist_indices = {indices[std::max<int>(i - 1, size_t(0))], indices[i], indices[i + 1], indices[std::min(i + 2, indices.size() - 1)]};
                        attribute_bounds = {active_ordered_attributes[std::max(i - 1, size_t(0))].get().bounds->read(), active_ordered_attributes[i].get().bounds->read(), active_ordered_attributes[i + 1].get().bounds->read(), active_ordered_attributes[std::min(i + 2, indices.size() - 1)].get().bounds->read()};
                        std::vector<uint32_t> ordering(hist_indices.size());
                        std::iota(ordering.begin(), ordering.end(), 0);
                        std::sort(ordering.begin(), ordering.end(), [&](uint32_t a, uint32_t b){return hist_indices[a] < hist_indices[b];});
                        pc.a_axis = i - 1 + ordering[0];
                        pc.b_axis = i - 1 + ordering[1];
                        pc.c_axis = i - 1 + ordering[2];
                        pc.d_axis = i - 1 + ordering[3];
                        pc.a_size = bin_sizes[ordering[0]];
                        pc.b_size = bin_sizes[ordering[1]];
                        pc.c_size = bin_sizes[ordering[2]];
                        pc.d_size = bin_sizes[ordering[3]];
                        pc.line_verts = _spline_resolution;
                    }
                    else{
                        hist_indices = {indices[i], indices[i + 1]};
                        attribute_bounds = {active_ordered_attributes[i].get().bounds->read(), active_ordered_attributes[i + 1].get().bounds->read()};
                        pc.a_axis = indices[i] < indices[i + 1] ? i + 1 : i;
                        pc.b_axis = indices[i] < indices[i + 1] ? i : i + 1;
                        pc.a_size = bin_sizes[0];
                        pc.b_size = bin_sizes[1];
                        pc.line_verts = 2;
                    }
                    // getting the correct hist information
                    bool is_max_hist = dl.priority_render.read();
                    pc.priority_rendering = dl.priority_render.read();
                    std::string id = util::histogram_registry::get_id_string(hist_indices, bin_sizes, attribute_bounds, false, is_max_hist);
                    {
                        auto hist_access = drawlist.histogram_registry.const_access();
                        if(!hist_access->name_to_registry_key.contains(id) || !hist_access->gpu_buffers.contains(id)){
                            if(logger.logging_level >= logging::level::l_4)
                                logger << logging::warning_prefix << " Missing histogram (" << id << "). Nothing rendered for subplot " << indices[i] << "|" << indices[i + 1] << logging::endl;
                            continue;
                        }
                        pc.histogram_address = util::vk::get_buffer_address(hist_access->gpu_buffers.at(id));
                    }
                    if(dl.drawlist_read().priority_indices.contains(id))
                        pc.ordering_address = util::vk::get_buffer_address(dl.drawlist_read().priority_indices.at(id));
                    pc.color = dl.appearance->read().color;
                    vkCmdPushConstants(_render_commands.back(), pipeline_info.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                    vkCmdDraw(_render_commands.back(), lines_amt * (pc.line_verts - 1) * 2, 1, 0, 0);
                }
                break;
            }
            default:
                throw std::runtime_error{"pipelines::parallel_coordinates_renderer::render(...) Rendering for rendering " + std::string(structures::parallel_coordinates_renderer::render_type_names[out_specs.render_typ]) + " not implemented"};
            }
            

            cur_offset += cur_batch_size;
            cur_batch_lines += cur_batch_size;
            vkCmdEndRenderPass(_render_commands.back());
            if(cur_batch_lines >= batch_size){
                // dispatching command buffer
                _post_render_commands(_render_commands.back());
                _render_commands.push_back(util::vk::create_begin_command_buffer(_command_pool));
                cur_batch_lines = 0;
            }
        } while(cur_offset < data_size);
    }
    // histogram rendering
    using histogram_type = workbenches::parallel_coordinates_workbench::histogram_type;
    if(info.workbench.setting.read().hist_type != histogram_type::none){
        // active histogram count for histogram width calculation
        int active_histogram_count{};
        for(const auto& dl: info.workbench.drawlist_infos.read()){
            if(dl.appearance->read().show_histogram)
                ++active_histogram_count;
        }
        float histogram_distance = (2. - info.workbench.setting.read().histogram_width) / (active_ordered_attributes.size() - 1);
        float drawlist_histogram_width = info.workbench.setting.read().histogram_width / active_histogram_count;
        output_specs out_specs{
            info.workbench.plot_data.read().image_view,
            info.workbench.plot_data.read().image_format, 
            info.workbench.plot_data.read().image_samples, 
            info.workbench.plot_data.read().width, 
            info.workbench.plot_data.read().height, 
            structures::parallel_coordinates_renderer::render_type::histogram, 
            {}
        };
        auto pipeline_info = get_or_create_pipeline(out_specs);

        _pre_render_commands(_render_commands.back(), out_specs, clear_framebuffer, info.workbench.setting.read().plot_background);
        clear_framebuffer = false;

        float histogram_offset = 0;
        for(const auto& dl_info: util::rev_iter(info.workbench.drawlist_infos.read())){
            if(!dl_info.appearance->read().show_histogram)
                continue;

            const auto& ds = dl_info.dataset_read();
            const auto active_indices = util::data::active_attribute_refs_to_indices(active_ordered_attributes, ds.attributes);
            push_constants_hist_frag pc_frag{};
            pc_frag.bin_count = info.workbench.plot_data.read().height;
            pc_frag.blur_radius = info.workbench.setting.read().histogram_blur_width;
            pc_frag.color = dl_info.appearance->read().color;
            pc_frag.mapping_type = static_cast<uint32_t>(info.workbench.setting.read().hist_type);
            for(int i: util::size_range(active_indices)){
                float x_base = -1. + i * histogram_distance + histogram_offset;

                uint32_t index = active_indices[i];
                int bin_size = info.workbench.plot_data.read().height;
                std::string hist_id = util::histogram_registry::get_id_string(index, bin_size, active_ordered_attributes[i].get().bounds->read(), false, false);
                {
                    auto hist_access = dl_info.drawlist_read().histogram_registry.const_access();
                    if(!hist_access->name_to_registry_key.contains(hist_id) || !hist_access->gpu_buffers.contains(hist_id)){
                        if(logger.logging_level >= logging::level::l_4)
                            logger << logging::warning_prefix << " Missing histogram (" << hist_id << "). No histogram for drawlist " << dl_info.drawlist_id << " on " << dl_info.drawlist_read().dataset_read().attributes[index].display_name << logging::endl;
                        continue;
                    }
                    pc_frag.histogram_address = util::vk::get_buffer_address(hist_access->gpu_buffers.at(hist_id));
                }
                push_constants_hist_vert pc_vert{};
                pc_vert.bounds = {x_base, -1, x_base + drawlist_histogram_width, 1};
                vkCmdPushConstants(_render_commands.back(), pipeline_info.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc_vert), &pc_vert);
                vkCmdPushConstants(_render_commands.back(), pipeline_info.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(pc_vert), sizeof(pc_frag), &pc_frag);
                vkCmdDraw(_render_commands.back(), 4, 1, 0, 0);
            }

            histogram_offset += drawlist_histogram_width;
        }
        vkCmdEndRenderPass(_render_commands.back());
    }
    if(clear_framebuffer){
        output_specs out_specs{
            info.workbench.plot_data.read().image_view,
            info.workbench.plot_data.read().image_format, 
            info.workbench.plot_data.read().image_samples, 
            info.workbench.plot_data.read().width, 
            info.workbench.plot_data.read().height, 
            structures::parallel_coordinates_renderer::render_type::polyline_spline,
            structures::data_type_t::float_t
        };
        auto pipeline_info = get_or_create_pipeline(out_specs);

        _pre_render_commands(_render_commands[0], out_specs, clear_framebuffer, info.workbench.setting.read().plot_background);
        vkCmdEndRenderPass(_render_commands[0]);
    }

    // committing last command buffer
    _post_render_commands(_render_commands.back(), _render_fence);    
}
}