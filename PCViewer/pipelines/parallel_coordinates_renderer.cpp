#include "parallel_coordinates_renderer.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <file_util.hpp>
#include <array>
#include <parallel_coordinates_workbench.hpp>
#include <array_struct.hpp>

namespace pipelines
{
parallel_coordinates_renderer::parallel_coordinates_renderer() 
{
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.graphics_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _render_fence = util::vk::create_fence(fence_info);
}

const parallel_coordinates_renderer::pipeline_data& parallel_coordinates_renderer::get_or_create_pipeline(const output_specs& output_specs){
    if(!_pipelines.contains(output_specs)){
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

        pipeline_data pipe_data{};
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
        auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo({}, util::memory_view(push_constant_range));
        pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

        // pipeline creation
        auto pipeline_rasterizer = util::vk::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_LINE, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
        pipeline_rasterizer.lineWidth = 1;

        auto pipeline_color_blend_attachment = util::vk::initializers::pipelineColorBlendAttachmentState(0xf, VK_TRUE);
        pipeline_color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	    pipeline_color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	    pipeline_color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
	    pipeline_color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	    pipeline_color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	    pipeline_color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

        auto pipeline_color_blend = util::vk::initializers::pipelineColorBlendStateCreateInfo(pipeline_color_blend_attachment);

        auto pipeline_viewport = util::vk::initializers::pipelineViewportStateCreateInfo(1, 1);

        std::vector<VkDynamicState> dynamic_states;
        auto pipeline_dynamic_states = util::vk::initializers::pipelineDynamicStateCreateInfo(dynamic_states); 

        auto pipeline_depth_stencil = util::vk::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

        auto pipeline_multi_sample = util::vk::initializers::pipelineMultisampleStateCreateInfo(output_specs.sample_count);
        
        auto vertex_input_binding = util::vk::initializers::vertexInputBindingDescription(0, sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX);
        auto vertex_input_attribute = util::vk::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32_SFLOAT, 0);
        auto pipeline_vertex_state = util::vk::initializers::pipelineVertexInputStateCreateInfo(vertex_input_binding, vertex_input_attribute);

        switch(output_specs.render_typ){
        case structures::parallel_coordinates_renderer::render_type::polyline:{
            VkShaderModule vertex_module = util::vk::create_shader_module(vertex_shader_path); 
            VkShaderModule fragment_module = util::vk::create_shader_module(fragment_shader_path);  

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module);
            shader_infos[1] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module);

            auto pipeline_input_assembly = util::vk::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, 0, VK_TRUE);

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
        case structures::parallel_coordinates_renderer::render_type::spline:{
            VkShaderModule vertex_module = util::vk::create_shader_module(vertex_shader_path); 
            VkShaderModule geometry_module = util::vk::create_shader_module(geometry_shader_path);
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
            vkDestroyShaderModule(globals::vk_context.device, geometry_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);
            break;
        }
        case structures::parallel_coordinates_renderer::render_type::large_vis_lines:{
            VkShaderModule vertex_module = util::vk::create_shader_module(large_vis_vertex_shader_path); 
            VkShaderModule geometry_module = util::vk::create_shader_module(geometry_shader_path);
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
            vkDestroyShaderModule(globals::vk_context.device, geometry_module, globals::vk_context.allocation_callbacks);
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
        
        auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, byte_size);
        auto allocation_info = util::vma::initializers::allocationCreateInfo();
        _attribute_info_buffer = util::vk::create_buffer(buffer_info, allocation_info);
    }
    return _attribute_info_buffer;
}

parallel_coordinates_renderer& parallel_coordinates_renderer::instance(){
    static parallel_coordinates_renderer renderer;
    return renderer;
}

void parallel_coordinates_renderer::render(const render_info& info){
    struct pipeline_uniform_infos{
        int test;
    };

    output_specs out_specs{info.workbench.plot_width, info.workbench.plot_height, info.workbench.plot_image_samples, info.workbench.plot_image_format, info.workbench.render_type, info.workbench.plot_image_view}; 
    auto pipeline_info = get_or_create_pipeline(out_specs);

    structures::dynamic_struct<pipeline_uniform_infos, ImVec4> pipeline_uniforms(info.workbench.attributes.size());
    auto attribute_infos = get_or_resize_info_buffer(pipeline_uniforms.data().byteSize());
    globals::vk_context.upload_to_staging_buffer(pipeline_uniforms.data());

    auto res = vkWaitForFences(globals::vk_context.device, 1, &_render_fence, VK_TRUE, 1e10); util::check_vk_result(res);  // 10 seconds waiting
    if(_render_commands)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_render_commands);
    _render_commands = util::vk::create_begin_command_buffer(_command_pool);

    util::vk::end_commit_command_buffer(_render_commands, globals::vk_context.graphics_queue, info.wait_semaphores, {}, info.signal_semaphores, _render_fence);
}
}