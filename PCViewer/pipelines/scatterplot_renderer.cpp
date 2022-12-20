#include "scatterplot_renderer.hpp"
#include <vma_initializers.hpp>
#include <global_descriptor_set_util.hpp>

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
            logger << "[info] parallel_coordinates_renderer::get_or_create_pipeline() creating new pipeline for output_specs " << ss.str() << logging::endl;
        }

        if(_pipelines.size() > max_pipeline_count){
            auto [pipeline, time] = *std::min_element(_pipeline_last_use.begin(), _pipeline_last_use.end(), [](const auto& l, const auto& r){return l.second < r.second;});
            VkPipeline pipe = pipeline; // needed for msvc and clang on windows...
            auto [key, val] = *std::find_if(_pipelines.begin(), _pipelines.end(), [&](const auto& e){return e.second.pipeline == pipe;});
            util::vk::destroy_pipeline(val.pipeline);
            util::vk::destroy_pipeline_layout(val.pipeline_layout);
            util::vk::destroy_render_pass(val.render_pass);
            _pipeline_last_use.erase(pipeline);
            _pipelines.erase(key);
        }

        pipeline_data& pipe_data = _pipelines[output_specs];
        // creating the rendering buffers  -------------------------------------------------------------------------------
        // output image after multisample reduction (already given by parallel coordinates workbench)

        // multisample image
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            if(_multisample_image.image && (_multisample_image.format != output_specs.format || _multisample_image.spp != output_specs.sample_count || _multisample_image.width != output_specs.width)){
                util::vk::destroy_image(_multisample_image.image);
                util::vk::destroy_image_view(_multisample_image.image_view);
            }
            if(!_multisample_image.image){
                auto image_info = util::vk::initializers::imageCreateInfo(output_specs.format, {output_specs.width, output_specs.width, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_TYPE_2D, 1, 1, output_specs.sample_count);
                auto allocation_info = util::vma::initializers::allocationCreateInfo();
                std::tie(_multisample_image.image, _multisample_image.image_view) = util::vk::create_image_with_view(image_info, allocation_info);

                // updating the image layout
                auto image_barrier = util::vk::initializers::imageMemoryBarrier(_multisample_image.image.image, VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, {}, {}, {}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                util::vk::convert_image_layouts_execute(image_barrier);
            }
            _multisample_image.format = output_specs.format;
            _multisample_image.spp = output_specs.sample_count;
            _multisample_image.width = output_specs.width;
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
            break;
        }
        case structures::scatterplot_wb::data_source_t::hsitogram_t:{
            // pipeline layout creation
            auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants_large_vis), 0);
            assert(globals::descriptor_sets.contains(util::global_descriptors::heatmap_descriptor_id));      // the iron map has to be already created before the pipeliens are created
            auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo(globals::descriptor_sets[util::global_descriptors::heatmap_descriptor_id]->layout, util::memory_view(push_constant_range));
            pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

            auto vertex_module = util::vk::create_scoped_shader_module(vertex_shader_path);
            auto fragment_module = util::vk::create_scoped_shader_module(fragment_shader_path);  

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, *vertex_module, &specialization_info),
                util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, *fragment_module)
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
            break;
        }
        default:
            throw std::runtime_error("scatterplot_renderer::get_or_create_pipeline() data_source(" + std::to_string(int(output_specs.data_source)) + ") not yet implemented.");
        }
    }
    _pipeline_last_use[_pipelines[output_specs].pipeline] = std::chrono::system_clock::now();
    return _pipelines[output_specs];
}

VkFramebuffer scatterplot_renderer::_get_or_create_framebuffer(VkRenderPass render_pass, uint32_t width, VkImageView dst_image, VkImageView multisample_image){
    if(!_framebuffers.contains({dst_image, multisample_image})){
        // free framebuffers if too many are allocated
        if(_framebuffers.size() > max_framebuffer_count){
            auto [framebuffer, time] = *std::min_element(_framebuffer_last_use.begin(), _framebuffer_last_use.end(), [](const auto& l, const auto& r){return l.second < r.second;});
            VkFramebuffer frameb = framebuffer; // needed for msvs and clang to insert structured binding into lambda
            auto [key, val] = *std::find_if(_framebuffers.begin(), _framebuffers.end(), [&](const auto& e){return e.second == frameb;});
            util::vk::destroy_framebuffer(framebuffer);
            _framebuffer_last_use.erase(framebuffer);
            _framebuffers.erase(key);
        }

        // creating the framebuffer for current image views
        std::vector<VkImageView> image_views{dst_image};
        if(multisample_image)
            image_views.push_back(multisample_image);
        auto framebuffer_info = util::vk::initializers::framebufferCreateInfo(render_pass, image_views, width, width, 1);
        _framebuffers[{dst_image, multisample_image}] = util::vk::create_framebuffer(framebuffer_info);
    }
    return _framebuffers[{dst_image, multisample_image}];
}

scatterplot_renderer& scatterplot_renderer::instance(){
    static scatterplot_renderer renderer;
    return renderer;
}

void scatterplot_renderer::render(const render_info& info){
    // TODO: implement
}
}