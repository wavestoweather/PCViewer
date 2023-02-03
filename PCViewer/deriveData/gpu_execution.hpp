#pragma once
#include <shader_compiler.hpp>
#include <vk_util.hpp>
#include <logger.hpp>
#include <sstream>
#include <charconv>
#include <robin_hood.h>
#include <variant>
#include <fast_float.h>
#include "gpu_instructions.hpp"

namespace deriveData{
struct pipeline_info{
    VkPipeline          pipeline;
    VkPipelineLayout    layout;
    // maybe additional info
};
using data_storage = std::variant<uint32_t, size_t, float>;
inline data_storage extract_data_storage(std::string_view address){
    data_storage ret;
    switch(address[0]){
    case 'l':
        ret = uint32_t{};
        std::from_chars(&*(address.begin() + 1), &*address.end(), std::get<uint32_t>(ret));
        break;
    case 'g':
        ret = size_t{};
        std::from_chars(&*(address.begin() + 1), &*address.end(), std::get<size_t>(ret));
        break;
    case 'c':
        ret = float{};
        fast_float::from_chars(&*(address.begin() + 1), &*address.end(), std::get<float>(ret));
        break;
    }
    return ret;
}

inline std::tuple<std::vector<data_storage>, std::vector<data_storage>> extract_input_output_indices(std::string_view line){
    std::vector<data_storage> input_indices; 
    std::vector<data_storage> output_indices;
    std::string_view inputs; util::getline(line, inputs, ' '); util::getline(line, inputs, ' '); inputs = inputs.substr(inputs.find('[')); inputs = inputs.substr(0, inputs.find(']'));
    std::string_view element;
    while(util::getline(inputs, element, ',')){
        input_indices.emplace_back(extract_data_storage(element));
    }
    std::string_view outputs; util::getline(line, outputs, ' '); outputs = outputs.substr(outputs.find('[')); outputs = outputs.substr(0, outputs.find(']'));
    while(util::getline(outputs, element, ',')){
        output_indices.emplace_back(extract_data_storage(element));
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

    struct data_state_t{
        std::vector<uint32_t> cur_dimension_sizes;          // if the dimensionsizes are getting bigger -> store, barrier and reload, if the dimensionsize are getting smaller -> reduction store (atomic store) and reload,
        robin_hood::unordered_set<uint64_t> loaded_data;
        std::vector<uint64_t> storage_data_pointers;
    } data_state;

    // creating shader codes and converting them to pipeliens
    std::vector<pipeline_info>  pipelines;
    std::stringstream           header;
    std::stringstream           body;
    size_t                      storage_size{};
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
        storage_size = std::max(storage_size, input_indices.size());
        storage_size = std::max(storage_size, output_indices.size());
        switch(operation){
        case op_codes::none:
            break;
        case op_codes::load:
            body << "vec array = vec(uint64_t(" << std::get<size_t>(input_indices[0]) << "));";
            break;
        case op_codes::store:
            body << "vec array = vec(uint64_t(" << std::get<size_t>(output_indices[0]) << "));\n";
            body << "array[gl_GlobalInvocationID.x} = storage[" << std::get<uint32_t>(input_indices[0]) << "];\n";
            break;
        case op_codes::pipeline_barrier:
            body << "}\n";
            header << "float storage[" << storage_size << "];" << body.str();
            pipelines.emplace_back(create_pipeline(header.str()));
            header.clear(); body.clear();
            break;
        case op_codes::one_vec:
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = 1;\n";
            break;
        case op_codes::zero_vec:
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = 0;\n";
            break;
        case op_codes::rand_vec:
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = random_float();\n";
            break;
        case op_codes::iota_vec:
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = float(gl_GlobalInvocationId.x);\n";
            break;
        case op_codes::copy:
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = storage[" << std::get<uint32_t>(input_indices[0]) << "];\n";
            break;
        case op_codes::sum:
            // TODO additional data, such as pre factors
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = ";
            for(auto&& [e, last]: util::last_iter(input_indices)){
                body << "storage[" << std::get<uint32_t>(e) << "]";
                if(!last) body << " + ";
            }
            body << ";\n";
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