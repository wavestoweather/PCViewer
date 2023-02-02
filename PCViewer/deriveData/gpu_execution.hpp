#pragma once
#include <shader_compiler.hpp>
#include <vk_util.hpp>
#include <logger.hpp>
#include <sstream>
#include <charconv>
#include "gpu_instructions.hpp"

namespace deriveData{
struct pipeline_info{
    VkPipeline          pipeline;
    VkPipelineLayout    layout;
    // maybe additional info
};
inline std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> extract_input_output_indices(std::string_view line){
    std::vector<uint32_t> input_indices; 
    std::vector<uint32_t> output_indices;
    std::string_view inputs; util::getline(line, inputs, ' '); util::getline(line, inputs, ' '); inputs = inputs.substr(inputs.find('[')); inputs = inputs.substr(0, inputs.find(']'));
    std::string_view element;
    while(util::getline(inputs, element, ',')){
        input_indices.emplace_back();
        std::from_chars(&*element.begin(), &*element.end(), input_indices.back());
    }
    std::string_view outputs; util::getline(line, outputs, ' '); outputs = outputs.substr(outputs.find('[')); outputs = outputs.substr(0, outputs.find(']'));
    while(util::getline(outputs, element, ',')){
        output_indices.emplace_back();
        std::from_chars(&*element.begin(), &*element.end(), output_indices.back());
    }
    return {std::move(input_indices), std::move(output_indices)};
}
inline std::vector<pipeline_info> create_gpu_pipelines(std::string_view instructions){
    auto create_pipeline = [](const std::string& code){
        if(logger.logging_level >= logging::level::l_5)
            logger << "deriveData()::create_gpu_pipelines() New pipeline created with code:\n" << code << logging::endl;
        auto spir_v = util::shader_compiler::compile(code);
        return pipeline_info{};
    };

    // creating shader codes and converting them to pipeliens
    std::vector<pipeline_info>  pipelines;
    std::stringstream           header;
    std::stringstream           body;
    uint32_t                    storage_size{};
    std::string_view            line;
    while(util::getline(instructions, line)){
        // hader
        if(header.str().empty()){
            header << R"(
                #version 450

                layout(push_constant) uniform PCs{
                    uint32_t rand_seed;
                };

                layout(local_size_x = 128) in;
            )";
            body << R"(
                float random_float(){
                    return .0f; // TODO
                }

                void main(){
            )";
        }

        // instrucction decoding
        std::stringstream line_stream{std::string(line.substr(0, line.find(' ')))};
        op_codes operation;
        line_stream >> operation;
        // decode storage fields
        auto [input_indices, output_indices] = extract_input_output_indices(line);
        storage_size = std::max(storage_size, input_indices | util::max());
        storage_size = std::max(storage_size, output_indices | util::max());
        switch(operation){
        case op_codes::none:
            break;
        case op_codes::load:
            break;
        case op_codes::store:
            break;
        case op_codes::pipeline_barrier:
            body << "}\n";
            header << "float storage[" << storage_size << "];" << body.str();
            pipelines.emplace_back(create_pipeline(header.str()));
            header.clear(); body.clear();
            break;
        case op_codes::one_vec:
            body << "storage[" << output_indices[0] << "] = 1;\n";
            break;
        case op_codes::zero_vec:
            body << "storage[" << output_indices[0] << "] = 0;\n";
            break;
        case op_codes::rand_vec:
            body << "storage[" << output_indices[0] << "] = 0;\n";
            break;
        case op_codes::iota_vec:
            break;
        case op_codes::copy:
            break;
        case op_codes::sum:
            break;
        default:
            assert(false && "Unimplemented operation");
        }
    }
    body << "}\n";
    header << "float storage[" << storage_size << "];" << body.str();
    pipelines.emplace_back(create_pipeline(header.str()));
    return pipelines;
}
}