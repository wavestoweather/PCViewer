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
    size_t              amt_of_threads;
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
inline std::tuple<std::vector<uint32_t>, std::vector<std::vector<uint32_t>>> extract_input_output_dimensions(std::string_view line){
    std::string_view dimension_info = (line | util::slice('('))[1];
    std::vector<uint32_t> dimension_sizes;
    std::vector<std::vector<uint32_t>> dimension_indices_v;
    std::string_view dimensions = (dimension_info.substr(1) | util::slice(']'))[0];
    for(auto size: dimensions | util::slice(',')){
        dimension_sizes.emplace_back();
        std::from_chars(&*size.begin(), &*size.end(), dimension_sizes.back());
    }
    auto tmp = dimension_info | util::slice('[');
    for(auto dimension_indices: util::subrange(tmp.begin() + 4, tmp.end())){
        dimension_indices_v.emplace_back();
        auto dim_inds = (dimension_indices | util::slice(']'))[0];
        for(auto ind: dim_inds | util::slice(',')){
            dimension_indices_v.back().emplace_back();
            std::from_chars(&*ind.begin(), &*ind.end(), dimension_indices_v.back().back());
        }
    }
    return {std::move(dimension_sizes), std::move(dimension_indices_v)};
}
struct create_gpu_pipelines_result{
    std::vector<pipeline_info> pipelines;
};
inline std::vector<pipeline_info> create_gpu_pipelines(std::string_view instructions){
    auto create_pipeline = [](const std::string& code){
        if(logger.logging_level >= logging::level::l_5)
            logger << "deriveData()::create_gpu_pipelines() New pipeline created with code:\n" << code << logging::endl;
        auto spir_v = util::shader_compiler::compile(code);
        return pipeline_info{};
    };

    struct data_state_t{
        std::vector<uint32_t> cur_dimension_sizes;          // if the dimensionsizes are getting bigger -> store, barrier and reload, if the dimensionsize are getting smaller -> reduction store (atomic store) and reload,
        std::vector<uint32_t> cur_dimension_indices;        // dimension indices // hold information about the 
        bool                  all_same_data_layout{true};   // indicates that all data inputs have the same data layout (useful for easier readout)
    } data_state;

    // creating shader codes and converting them to pipeliens
    std::vector<pipeline_info>  pipelines;
    std::stringstream           header;
    std::stringstream           body;
    size_t                      storage_size{};
    for(std::string_view line: instructions | util::slice('\n')){
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
                uint _random_state = gl_GlobalInvocationID.x;
                float random_float(){
                    _random_state ^= _random_state << 21;
                    _random_state ^= _random_state >> 35;
                    _random_state ^= _random_state << 4;
                    return float(_random_state) / float(0xffffffff);
                }

                void main(){
            )";
        }

        // instrucction decoding
        std::stringstream line_stream{std::string(line.substr(0, line.find(' ')))};
        op_codes operation;
        line_stream >> operation;
        const auto [dim_sizes, dim_indices] = extract_input_output_dimensions(line);
        const auto [input_indices, output_indices] = extract_input_output_indices(line);
        storage_size = std::max(storage_size, input_indices.size());
        storage_size = std::max(storage_size, output_indices.size());
        // check dimension consistency
        for(const auto& i: dim_indices){
            for(uint32_t dim: i){
                if(!(data_state.cur_dimension_indices | util::contains(dim)))
                    data_state.cur_dimension_indices.emplace_back(dim);
            }
        }
        const bool same_before = data_state.all_same_data_layout;
        for(uint32_t i: util::i_range(dim_indices.size() - 1))
            data_state.all_same_data_layout &= dim_indices[i] == dim_indices[i + 1];
        assert(!same_before || same_before == data_state.all_same_data_layout);
        assert(data_state.cur_dimension_sizes.empty() || data_state.cur_dimension_sizes == dim_sizes);
        data_state.cur_dimension_sizes = dim_sizes;
        
        // decode storage fields
        switch(operation){
        case op_codes::none:
            break;
        case op_codes::load:
            for(auto&& [input_index, i]: util::enumerate(input_indices))
                body << "vec array"<< i <<" = vec(uint64_t(" << std::get<size_t>(input_index) << "));";
            if(data_state.all_same_data_layout){
                for(auto&& [out_index, i]: util::enumerate(output_indices))
                    body << "storage[" << std::get<uint32_t>(out_index) << "] = array" << i << "[gl_GlobalInvocationID.x];\n";
            }
            else{
                body << "float dimension_indices["<< data_state.cur_dimension_sizes.size() <<"]; uint i = gl_GlobalInvocationID.x;\n";
                for(size_t i: util::rev_size_range(data_state.cur_dimension_sizes))
                    body << "dimension_indices[" << i << "] = i % " << data_state.cur_dimension_sizes[i] << "; i /= " << data_state.cur_dimension_sizes[i] << ";\n";
                body << "uint dim_mult;\n";
                for(auto&& [out_index, i]: util::enumerate(output_indices)){
                    std::string input_index_var("in" + std::to_string(i));
                    body << "uint " << input_index_var << " = 0; dim_mult = 1;\n";
                    for(size_t j: util::rev_size_range(dim_indices[i])){
                        body << input_index_var << " += dim_mult * dimension_indices[" << dim_indices[i][j] << "];";
                        if(j > 0)
                            body << "dim_mult *= " << dim_sizes[dim_indices[i][j]] << ";\n";
                    }
                    body << "storage[" << std::get<uint32_t>(out_index) << "] = array" << i << "[" << input_index_var << "];\n";
                }
            }
            break;
        case op_codes::store:
            for(auto&& [out_index, i]: util::enumerate(output_indices)){
                body << "vec array" << i << " = vec(uint64_t(" << std::get<size_t>(out_index) << "));\n";
                body << "array" << i << "[gl_GlobalInvocationID.x] = storage[" << std::get<uint32_t>(out_index) << "];\n";
            }
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
    if(storage_size)
    body << "}\n";
    header << "float storage[" << storage_size << "];" << body.str();
    pipelines.emplace_back(create_pipeline(header.str()));
    return pipelines;
}
}