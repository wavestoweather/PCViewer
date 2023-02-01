#pragma once
#include <shader_compiler.hpp>
#include <vk_util.hpp>
#include <logger.hpp>

namespace deriveData{
struct pipeline_info{
    VkPipeline          pipeline;
    VkPipelineLayout    layout;
    // maybe additional info
};
inline std::vector<pipeline_info> create_gpu_pipelines(std::string_view instructions){
    auto create_pipeline = [](const std::string& code){
        if(logger.logging_level >= logging::level::l_5)
            logger << "deriveData()::create_gpu_pipelines() New pipeline created with code:\n" << code << logging::endl;
        auto spir_v = util::shader_compiler::compile(code);
        return pipeline_info{};
    };

    // creating shader codes and converting them to pipeliens
    std::vector<pipeline_info> pipelines;
    std::stringstream cur_shader;
    std::string_view line;
    while(util::getline(instructions, line)){
        // hader
        if(cur_shader.str().empty()){
            cur_shader << R"(
                #version 450

                layout(local_size_x = 128) in;

                void main(){
            )";
        }
    }
    cur_shader << "}\n";
    pipelines.emplace_back(create_pipeline(cur_shader.str()));
    return pipelines;
}
}