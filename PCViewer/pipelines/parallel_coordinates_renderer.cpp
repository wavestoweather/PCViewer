#include "parallel_coordinates_renderer.hpp"
#include <vk_initializers.hpp>

namespace pipelines
{
parallel_coordinates_renderer::parallel_coordinates_renderer() 
{
    
}

const parallel_coordinates_renderer::pipeline_data& parallel_coordinates_renderer::get_or_create_pipeline(const output_specs& output_specs){
    if(!_pipelines.contains(output_specs)){
        // creating the rendering pipeline

    }
    return _pipelines[output_specs];
}

parallel_coordinates_renderer& parallel_coordinates_renderer::instance(){
    static parallel_coordinates_renderer renderer;
    return renderer;
}
}