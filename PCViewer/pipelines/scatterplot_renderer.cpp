#include "scatterplot_renderer.hpp"
#include <vma_initializers.hpp>
#include <global_descriptor_set_util.hpp>
#include <scatterplot_workbench.hpp>
#include <histogram_registry_util.hpp>

namespace pipelines{
scatterplot_renderer::scatterplot_renderer()
{
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _render_fence = util::vk::create_fence(fence_info);
}

const scatterplot_renderer::pipeline_data& scatterplot_renderer::_get_or_create_pipeline(const output_specs& output_specs){
    if(!_pipelines.contains(output_specs)){
        if(logger.logging_level >= logging::level::l_4){
            std::stringstream ss; ss << util::memory_view<const uint32_t>(util::memory_view(output_specs));
            logger << "[info] scatterplot_renderer::get_or_create_pipeline() creating new pipeline for output_specs " << ss.str() << logging::endl;
        }

        if(_pipelines.size() > max_pipeline_count){
            auto [pipeline, time] = *std::min_element(_pipeline_last_use.begin(), _pipeline_last_use.end(), [](const auto& l, const auto& r){return l.second < r.second;});
            VkPipeline pipe = pipeline; // needed for msvc and clang on windows...
            auto [key, val] = *std::find_if(_pipelines.begin(), _pipelines.end(), [&](const auto& e){return e.second.pipeline == pipe;});
            util::vk::destroy_pipeline(val.pipeline);
            util::vk::destroy_pipeline_layout(val.pipeline_layout);
            _pipeline_last_use.erase(pipeline);
            _pipelines.erase(key);
        }

        pipeline_data& pipe_data = _pipelines[output_specs];
        // creating the rendering pipeline -------------------------------------------------------------------------------

        // pipeline creation
        auto pipeline_rasterizer = util::vk::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_POINT, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipeline_rasterizer.lineWidth = 1;

        auto pipeline_color_blend_attachment = util::vk::initializers::pipelineColorBlendAttachmentStateStandardAlphaBlend();

        auto pipeline_color_blend = util::vk::initializers::pipelineColorBlendStateCreateInfo(pipeline_color_blend_attachment);

        auto pipeline_viewport = util::vk::initializers::pipelineViewportStateCreateInfo(1, 1);

        std::vector<VkDynamicState> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        auto pipeline_dynamic_states = util::vk::initializers::pipelineDynamicStateCreateInfo(dynamic_states); 

        auto pipeline_depth_stencil = util::vk::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

        auto pipeline_multi_sample = util::vk::initializers::pipelineMultisampleStateCreateInfo(output_specs.sample_count);

        auto pipeline_vertex_state = util::vk::initializers::pipelineVertexInputStateCreateInfo();//(vertex_input_binding, vertex_input_attribute);

        auto specialization_map_entry = util::vk::initializers::specializationMapEntry(0, 0, sizeof(output_specs.data_type));
        auto specialization_info = util::vk::initializers::specializationInfo(specialization_map_entry, util::memory_view(output_specs.data_type));

        switch(output_specs.data_source){
        case structures::scatterplot_wb::data_source_t::array_t:{
            // pipeline layout creation
            auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants), 0);
            assert(globals::descriptor_sets.contains(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout, util::memory_view(push_constant_range));
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            auto vertex_module = util::vk::create_scoped_shader_module(vertex_shader_path);
            auto fragment_module = util::vk::create_scoped_shader_module(fragment_shader_path);  

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, *vertex_module, &specialization_info);
            shader_infos[1] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, *fragment_module);

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, output_specs.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipeline(pipeline_create_info);
            break;
        }
        case structures::scatterplot_wb::data_source_t::histogram_t:{
            // pipeline layout creation
            auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants_large_vis), 0);
            assert(globals::descriptor_sets.contains(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout, util::memory_view(push_constant_range));
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            auto vertex_module = util::vk::create_scoped_shader_module(large_vis_vertex_shader_path);
            auto fragment_module = util::vk::create_scoped_shader_module(fragment_shader_path);  

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, *vertex_module, &specialization_info),
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, *fragment_module)
            };

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);

            auto pipeline_create_info = util::vk::initializers::graphicsPipelineCreateInfo(shader_infos, pipe_data.pipeline_layout, output_specs.render_pass);
            pipeline_create_info.pVertexInputState = &pipeline_vertex_state;
            pipeline_create_info.pInputAssemblyState = &pipeline_input_assembly;
            pipeline_create_info.pRasterizationState = &pipeline_rasterizer;
            pipeline_create_info.pColorBlendState = &pipeline_color_blend;
            pipeline_create_info.pMultisampleState = &pipeline_multi_sample;
            pipeline_create_info.pViewportState = &pipeline_viewport;
            pipeline_create_info.pDepthStencilState = &pipeline_depth_stencil;
            pipeline_create_info.pDynamicState = &pipeline_dynamic_states;

            pipe_data.pipeline = util::vk::create_graphics_pipeline(pipeline_create_info);
            break;
        }
        default:
            throw std::runtime_error("scatterplot_renderer::get_or_create_pipeline() data_source(" + std::to_string(int(output_specs.data_source)) + ") not yet implemented.");
        }
    }
    _pipeline_last_use[_pipelines[output_specs].pipeline] = std::chrono::system_clock::now();
    return _pipelines[output_specs];
}

const scatterplot_renderer::framebuffer_val& scatterplot_renderer::_get_or_create_framebuffer(const framebuffer_key& frambuffer_key){
    if(!_framebuffers.contains(frambuffer_key)){
        if(logger.logging_level >= logging::level::l_4){
            std::stringstream ss; ss << util::memory_view<const uint32_t>(util::memory_view(frambuffer_key));
            logger << "[info] scatterplot_renderer::_get_or_create_framebuffer() creating new pipeline for framebuffer_key " << ss.str() << logging::endl;
        }

        // destroying last use framebuffer
        if(_framebuffers.size() > max_framebuffer_count){
            auto [framebuffer, time] = *std::min_element(_framebuffer_last_use.begin(), _framebuffer_last_use.end(), [](const auto& l, const auto& r){return l.second < r.second;});
            VkFramebuffer frame = framebuffer; // needed for msvc and clang on windows...
            auto [key, val] = *std::find_if(_framebuffers.begin(), _framebuffers.end(), [&](const auto& e){return e.second.framebuffer == frame;});
            util::vk::destroy_image(val.image);
            util::vk::destroy_image(val.multisample_image);
            util::vk::destroy_image_view(val.image_view);
            util::vk::destroy_image_view(val.multisample_image_view);
            util::vk::destroy_render_pass(val.render_pass);
            util::vk::destroy_framebuffer(val.framebuffer);
            _framebuffer_last_use.erase(framebuffer);
            _framebuffers.erase(key);
        }

        auto& framb_val = _framebuffers[frambuffer_key];
        // creating rendering images
        auto image_info = util::vk::initializers::imageCreateInfo(frambuffer_key.format, {frambuffer_key.width, frambuffer_key.width, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_IMAGE_TYPE_2D);
        auto alloc_info = util::vma::initializers::allocationCreateInfo();
        std::tie(framb_val.image, framb_val.image_view) = util::vk::create_image_with_view(image_info, alloc_info);
        if(frambuffer_key.sample_counts != VK_SAMPLE_COUNT_1_BIT){
            image_info.samples = frambuffer_key.sample_counts;
            std::tie(framb_val.multisample_image, framb_val.multisample_image_view) = util::vk::create_image_with_view(image_info, alloc_info);
        }
        // updating the image layout
        std::vector<VkImageMemoryBarrier> image_barriers{util::vk::initializers::imageMemoryBarrier(framb_val.image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
            util::vk::initializers::imageMemoryBarrier(framb_val.image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)};
        util::vk::convert_image_layouts_execute(image_barriers);

        // creating new framebuffer and render pass
        std::vector<VkAttachmentDescription> attachments;
        VkAttachmentDescription attachment = util::vk::initializers::attachmentDescription(frambuffer_key.format);
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachments.push_back(attachment);
        if(frambuffer_key.sample_counts != VK_SAMPLE_COUNT_1_BIT){
            attachment.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            attachment.samples = frambuffer_key.sample_counts;
            attachments.push_back(attachment);
        }
        std::vector<VkAttachmentReference> attachment_references;
        VkAttachmentReference attachment_reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        attachment_references.push_back(attachment_reference);
        util::memory_view<VkAttachmentReference> resolve_reference{};
        if(frambuffer_key.sample_counts != VK_SAMPLE_COUNT_1_BIT){
            attachment_references.back().attachment = 1;    // normal rendering goes to attachment 1 in multisampling case (at index 1 multisample image is attached)
            attachment_reference.attachment = 0;
            resolve_reference = util::memory_view(attachment_reference);
        }
        auto subpass_description = util::vk::initializers::subpassDescription(VK_PIPELINE_BIND_POINT_GRAPHICS, {}, attachment_references, resolve_reference);

        auto render_pass_info = util::vk::initializers::renderPassCreateInfo(attachments, subpass_description);
        framb_val.render_pass = util::vk::create_render_pass(render_pass_info);

        std::vector<VkImageView> image_views{framb_val.image_view};
        if(frambuffer_key.sample_counts != VK_SAMPLE_COUNT_1_BIT)
            image_views.push_back(framb_val.multisample_image_view);
        auto framebuffer_info = util::vk::initializers::framebufferCreateInfo(framb_val.render_pass, image_views, frambuffer_key.width, frambuffer_key.width, 1);
        framb_val.framebuffer = util::vk::create_framebuffer(framebuffer_info);
    }
    return _framebuffers[frambuffer_key];
}

scatterplot_renderer& scatterplot_renderer::instance(){
    static scatterplot_renderer renderer;
    return renderer;
}

void scatterplot_renderer::render(const render_info& info){
    // getting the framebuffer
    framebuffer_key fb_key{
        info.workbench.settings.read().plot_format,
        util::vk::sample_count_to_flag_bits(info.workbench.settings.read().sample_count),
        info.workbench.settings.read().plot_width
    };
    const auto& framebuffer = _get_or_create_framebuffer(fb_key);

    if(!info.workbench.plot_list.read().empty()){
        auto res = vkWaitForFences(globals::vk_context.device, 1, &_render_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
        vkResetFences(globals::vk_context.device, 1, &_render_fence);

        if(_render_commands.size())
            vkFreeCommandBuffers(globals::vk_context.device, _command_pool, _render_commands.size(), _render_commands.data());
        _render_commands.clear();
    }

    // renderng all plots with all datasets
    for(const auto& [axis_pair, iter_pos]: util::pos_iter(info.workbench.plot_list.read())){
        const auto& plot_data = info.workbench.plot_datas.at(axis_pair);

        _render_commands.emplace_back(util::vk::create_begin_command_buffer(_command_pool));
        VkClearValue clear_value{};
        auto render_info = util::vk::initializers::renderPassBeginInfo(framebuffer.render_pass, framebuffer.framebuffer, {0, 0, fb_key.width, fb_key.width}, clear_value);
        vkCmdBeginRenderPass(_render_commands.back(), &render_info, {});
        VkViewport viewport{};
        viewport.width = fb_key.width;
        viewport.height = fb_key.width;
        viewport.maxDepth = 1;
        vkCmdSetViewport(_render_commands.back(), 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = {fb_key.width, fb_key.width};
        vkCmdSetScissor(_render_commands.back(), 0, 1, &scissor);
        for(const auto& [dl, dl_pos]: util::pos_iter(info.workbench.drawlist_infos.read())){
            if(dl_pos == util::iterator_pos::first){
                VkClearAttachment clear_value{};
                clear_value.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                VkClearRect clear_rect{};
                clear_rect.layerCount = 1;
                clear_rect.rect.extent = {fb_key.width, fb_key.width};
                vkCmdClearAttachments(_render_commands.back(), 1, &clear_value, 1, &clear_rect);
            }
            
            bool histogram_render = dl.templatelist_read().data_size > info.workbench.settings.read().large_vis_threshold;
            // getting the pipeline
            output_specs p_key{
                fb_key.format,
                fb_key.sample_counts,
                fb_key.width,
                dl.dataset_read().data_flags.data_type,
                histogram_render ? structures::scatterplot_wb::data_source_t::histogram_t: structures::scatterplot_wb::data_source_t::array_t,
                framebuffer.render_pass
            };
            const auto& pipeline = _get_or_create_pipeline(p_key);
            if(histogram_render){
                push_constants_large_vis pc{};
                {
                    std::array<int, 2> bin_sizes{int(p_key.width), int(p_key.width)};
                    auto hist_access = dl.drawlist_read().histogram_registry.const_access();
                    std::string histogram_id = util::histogram_registry::get_id_string(util::memory_view(&axis_pair.a, 2), bin_sizes, false, false);
                    if(!hist_access->name_to_registry_key.contains(histogram_id)){
                        if(logger.logging_level > logging::level::l_4)
                            logger << logging::warning_prefix << " scatterplot_renderer::render() Missing histogram for attributes " << info.workbench.attributes.read()[axis_pair.a].display_name << "|" << info.workbench.attributes.read()[axis_pair.b].display_name << " for drawwlist " << dl.drawlist_id << logging::endl;
                        continue;
                    }
                    pc.flip_axes = axis_pair.a > axis_pair.b;
                    pc.bin_size = p_key.width;
                    pc.form = static_cast<uint32_t>(dl.scatter_appearance.read().splat);
                    pc.radius = dl.scatter_appearance.read().radius;
                    pc.color = dl.appearance->read().color;
                    pc.counts_address = util::vk::get_buffer_address(hist_access->gpu_buffers.at(histogram_id));
                }
                vkCmdPushConstants(_render_commands.back(), pipeline.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            }
            else{
                push_constants pc{};
                pc.data_header_address = util::vk::get_buffer_address(dl.dataset_read().gpu_data.header);
                pc.index_buffer_address = util::vk::get_buffer_address(dl.templatelist_read().gpu_indices);
                pc.activation_bitset_address = util::vk::get_buffer_address(dl.drawlist_read().active_indices_bitset_gpu);
                pc.attribute_a = axis_pair.a;
                pc.attribute_b = axis_pair.b;
                pc.a_min = info.workbench.attributes.read()[axis_pair.a].bounds.read().min;
                pc.a_max = info.workbench.attributes.read()[axis_pair.a].bounds.read().max;
                pc.b_min = info.workbench.attributes.read()[axis_pair.b].bounds.read().min;
                pc.b_max = info.workbench.attributes.read()[axis_pair.b].bounds.read().max;
                pc.flip_axes = 1;   // always flip, as major axis is y axis
                pc.form = static_cast<uint32_t>(dl.scatter_appearance.read().splat);
                pc.radius = dl.scatter_appearance.read().radius;
                pc.color = dl.appearance->read().color;
                vkCmdPushConstants(_render_commands.back(), pipeline.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            }
            vkCmdBindPipeline(_render_commands.back(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
            vkCmdDraw(_render_commands.back(), histogram_render? fb_key.width * fb_key.width: dl.templatelist_read().data_size, 1, 0, 0);
        }
        vkCmdEndRenderPass(_render_commands.back());

        // image transfer to plot image
        const auto& plot_image = info.workbench.plot_datas.at(axis_pair).image;
        VkImageSubresourceRange subresource_range{}; subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; subresource_range.baseMipLevel = 0; subresource_range.levelCount = 1; subresource_range.baseArrayLayer = 0; subresource_range.layerCount = 1;
        //barriers.rendered_image = util::vk::initializers::imageMemoryBarrier(framebuffer.image.image, subresource_range, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        auto plot_image_barrier = util::vk::initializers::imageMemoryBarrier(plot_image.image, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdPipelineBarrier(_render_commands.back(), VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, {}, 0, {}, 1, &plot_image_barrier);

        VkImageCopy copy_region{}; copy_region.srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy_region.dstSubresource = copy_region.srcSubresource; copy_region.extent = {fb_key.width, fb_key.width, 1};
        vkCmdCopyImage(_render_commands.back(), framebuffer.image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, plot_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        //barriers.rendered_image = util::vk::initializers::imageMemoryBarrier(framebuffer.image.image, subresource_range, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        plot_image_barrier = util::vk::initializers::imageMemoryBarrier(plot_image.image, subresource_range, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        vkCmdPipelineBarrier(_render_commands.back(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 0, {}, 0, {}, 1, &plot_image_barrier);

        std::scoped_lock queue_lock(*globals::vk_context.graphics_mutex);
        util::vk::end_commit_command_buffer(_render_commands.back(), globals::vk_context.graphics_queue, {}, {}, {}, iter_pos == util::iterator_pos::last ? _render_fence: VkFence{});
    }
    if(!info.workbench.plot_list.read().empty()){
        auto res = vkWaitForFences(globals::vk_context.device, 1, &_render_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    }
}
}