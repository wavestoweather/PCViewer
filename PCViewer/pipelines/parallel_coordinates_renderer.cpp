#include "parallel_coordinates_renderer.hpp"
#include <vk_initializers.hpp>
#include <vma_initializers.hpp>
#include <vk_util.hpp>
#include <file_util.hpp>
#include <array>

namespace pipelines
{
parallel_coordinates_renderer::parallel_coordinates_renderer() 
{
    
}

const parallel_coordinates_renderer::pipeline_data& parallel_coordinates_renderer::get_or_create_pipeline(const output_specs& output_specs){
    if(!_pipelines.contains(output_specs)){
        pipeline_data pipe_data{};
        // creating the rendering buffers  -------------------------------------------------------------------------------
        // output image after multisample reduction
        auto image_info = util::vk::initializers::imageCreateInfo(output_specs.format, {output_specs.width, output_specs.height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR, VK_IMAGE_TYPE_2D);
        auto allocation_info = util::vma::initializers::allocationCreateInfo();
        std::tie(pipe_data.image, pipe_data.image_view) = util::vk::create_image_with_view(image_info, allocation_info);

        // multisample image
        if(output_specs.sample_count != VK_SAMPLE_COUNT_1_BIT){
            image_info = util::vk::initializers::imageCreateInfo(output_specs.format, {output_specs.width, output_specs.height, 1}, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR, VK_IMAGE_TYPE_2D);
        }

        // creating the rendering pipeline -------------------------------------------------------------------------------

        // pipeline layout creation
        auto push_constant_range = util::vk::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(push_constants), 0);
        auto layout_create = util::vk::initializers::pipelineLayoutCreateInfo({}, util::memory_view(push_constant_range));
        pipe_data.pipeline_layout = util::vk::create_pipeline_layout(layout_create);

        // pipeline creation
        switch(output_specs.render_type){
        case structures::parallel_coordinates_renderer::render_type::polyline:{
            VkShaderModule vertex_module; 
            VkShaderModule fragment_module; 

            auto vertex_bytes = util::read_file(vertex_shader_path);
            auto vertex_module_info = util::vk::initializers::shaderModuleCreateInfo(vertex_bytes);
            auto res = vkCreateShaderModule(globals::vk_context.device, &vertex_module_info, globals::vk_context.allocation_callbacks, &vertex_module); 
            util::check_vk_result(res);
            auto fragment_bytes = util::read_file(fragment_shader_path);
            auto fragment_module_info = util::vk::initializers::shaderModuleCreateInfo(fragment_bytes);
            res = vkCreateShaderModule(globals::vk_context.device, &fragment_module_info, globals::vk_context.allocation_callbacks, &fragment_module); 
            util::check_vk_result(res);

            std::array<VkPipelineShaderStageCreateInfo, 2> shader_infos{};
            shader_infos[0] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertex_module);
            shader_infos[1] = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_module);


            vkDestroyShaderModule(globals::vk_context.device, vertex_module, globals::vk_context.allocation_callbacks);
            vkDestroyShaderModule(globals::vk_context.device, fragment_module, globals::vk_context.allocation_callbacks);
            break;
        }
        case structures::parallel_coordinates_renderer::render_type::spline:

            break;
        case structures::parallel_coordinates_renderer::render_type::large_vis_lines:

            break;
        default:
            throw std::runtime_error("parallel_coordinates_renderer::get_or_create_pipeline() render_type " + (std::string)structures::parallel_coordinates_renderer::render_type_names[output_specs.render_type] + " not yet implemented.");
        }

    }
    return _pipelines[output_specs];
}

parallel_coordinates_renderer::parallel_coordinates_renderer() 
{
    
}

parallel_coordinates_renderer& parallel_coordinates_renderer::instance(){
    static parallel_coordinates_renderer renderer;
    return renderer;
}
}