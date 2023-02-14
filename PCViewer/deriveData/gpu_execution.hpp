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

#define workgroup_size 128
#define STR_IND(s) STR(s)
#define STR(s) #s

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
    std::string_view inputs; util::getline(line, inputs, ' '); util::getline(line, inputs, ' '); inputs = inputs.substr(inputs.find('[') + 1); inputs = inputs.substr(0, inputs.find(']'));
    std::string_view element;
    while(util::getline(inputs, element, ',')){
        input_indices.emplace_back(extract_data_storage(element));
    }
    std::string_view outputs; util::getline(line, outputs, ' '); outputs = outputs.substr(outputs.find('[') + 1); outputs = outputs.substr(0, outputs.find(']'));
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

// adds all the load store operations in a possibly optimal manner
// this also means, that the mapping from global to local address space is done here
// an example would be:
// op_vec_iota a b
// ->
// op_load a 0
// op_vec_iota 0
// op_store 0 b
// where a and b are global addresses, and 0 is the index into the local storage
inline std::string optimize_operations(const std::string& input){
    std::stringstream result_stream;
    for(std::string_view line: input | util::slice('\n')){
        const auto [input_indices, output_indices] = extract_input_output_indices(line);
        auto slice = (line | util::slice(' ')).begin();
        std::string_view op_code = *slice++;
        std::string_view input = *slice++, output = *slice, rest = slice.get_rest();
        std::stringstream internal_pos, internal_out_pos;
        internal_pos << '[';
        for(size_t i: util::size_range(input_indices)){
            internal_pos << 'l' << i;
            if(i != input_indices.size() - 1)
                internal_pos << ',';
        }
        internal_pos << "]";
        internal_out_pos << '[';
        for(size_t i: util::size_range(output_indices)){
            internal_out_pos << 'l' << i;
            if(i != input_indices.size() - 1)
                internal_out_pos << ',';
        }
        internal_out_pos << "]";
        // load
        result_stream << op_codes::load << ' ' << input << ' ' << internal_pos.str() << ' ' << rest << '\n';

        // main operation
        result_stream << op_code << ' ' << internal_pos.str() << ' ' << internal_out_pos.str() << ' ' << rest << '\n';

        // store operation
        result_stream << op_codes::store << ' ' << internal_out_pos.str() << ' ' << output << ' ' << rest << '\n';
    }
    return result_stream.str();
}

struct create_gpu_pipelines_result{
    std::vector<pipeline_info> pipelines;
};
inline std::vector<pipeline_info> create_gpu_pipelines(std::string_view instructions){
    struct data_state_t{
        std::vector<uint32_t> cur_dimension_sizes;          // if the dimensionsizes are getting bigger -> store, barrier and reload, if the dimensionsize are getting smaller -> reduction store (atomic store) and reload,
        std::vector<uint32_t> cur_dimension_indices;        // dimension indices // hold information about the 
        bool                  all_same_data_layout{true};   // indicates that all data inputs have the same data layout (useful for easier readout)
    } data_state;

    auto calc_thread_amt = [](const data_state_t& data_state){
        size_t size = 1;
        if(data_state.all_same_data_layout){
            for(auto i: data_state.cur_dimension_indices)
                size *= data_state.cur_dimension_sizes[i];
        }
        else{
            for(auto s: data_state.cur_dimension_sizes)
                size *= s;
        }
        return size;
    };
    auto create_pipeline = [](const std::string& code, size_t thread_amt){
        if(logger.logging_level >= logging::level::l_5)
            logger << "deriveData()::create_gpu_pipelines() New pipeline created with code:\n" << code << logging::endl;
        auto spir_v = util::shader_compiler::compile(code);
        pipeline_info ret{};
        ret.amt_of_threads = thread_amt;
        auto pipeline_layout_info = util::vk::initializers::pipelineLayoutCreateInfo();
        ret.layout = util::vk::create_pipeline_layout(pipeline_layout_info);
        auto shader_module = util::vk::create_scoped_shader_module(util::memory_view<const uint32_t>(spir_v));
        auto shader_stage_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(ret.layout, shader_stage_info);
        ret.pipeline = util::vk::create_compute_pipeline(pipeline_info);
        return ret;
    };


    // creating shader codes and converting them to pipelines
    std::vector<pipeline_info>  pipelines;
    std::stringstream           header;
    std::stringstream           body;
    size_t                      storage_size{};
    for(std::string_view line: instructions | util::slice('\n')){
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

        // header
        if(header.str().empty()){
            header << R"(
                #version 450
                #extension GL_EXT_buffer_reference2: require
                #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
                #extension GL_EXT_scalar_block_layout: require

                layout(buffer_reference, scalar) buffer vec{
                    float data[];
                };

                layout(push_constant) uniform PCs{
                    uint rand_seed;
                };

                layout(local_size_x = )" STR_IND(workgroup_size) R"() in;
            )";
            body << R"(
                uint _random_state = gl_GlobalInvocationID.x * 12 + (1 << 17) - 1;
                float random_float(){
                    _random_state ^= _random_state << 21;
                    _random_state ^= _random_state >> 35;
                    _random_state ^= _random_state << 4;
                    return float(_random_state) / float(uint(0xffffffff));
                }

                void main(){
                    if(gl_GlobalInvocationID.x >= )" << calc_thread_amt(data_state) << ") return;\n";
        }

        // loading the data

        // instruction decoding
        switch(operation){
        case op_codes::none:
            break;
        case op_codes::load:

            for(auto&& [input_index, i]: util::enumerate(input_indices))
                if(std::holds_alternative<size_t>(input_index))
                    body << "vec array"<< i <<" = vec(" << std::get<size_t>(input_index) << "ul);";
            if(data_state.all_same_data_layout){
                for(auto&& [out_index, i]: util::enumerate(output_indices)){
                    if(std::holds_alternative<float>(input_indices[i]))
                        body << "storage[" << std::get<uint32_t>(out_index) << "] = " << std::get<float>(input_indices[i]) << ";\n";
                    else
                        body << "storage[" << std::get<uint32_t>(out_index) << "] = array" << i << ".data[gl_GlobalInvocationID.x];\n";
                }
            }
            else{
                body << "float dimension_indices["<< data_state.cur_dimension_sizes.size() <<"]; uint i = gl_GlobalInvocationID.x;\n";
                for(size_t i: util::rev_size_range(data_state.cur_dimension_sizes))
                    body << "dimension_indices[" << i << "] = i % " << data_state.cur_dimension_sizes[i] << "; i /= " << data_state.cur_dimension_sizes[i] << ";\n";
                body << "uint dim_mult;\n";
                for(auto&& [out_index, i]: util::enumerate(output_indices)){
                    if(std::holds_alternative<float>(input_indices[i])){
                        body << "storage[" << std::get<uint32_t>(out_index) << "] = " << std::get<float>(input_indices[i]) << ";\n";
                        continue;
                    }
                    std::string input_index_var("in" + std::to_string(i));
                    body << "uint " << input_index_var << " = 0; dim_mult = 1;\n";
                    for(size_t j: util::rev_size_range(dim_indices[i])){
                        body << input_index_var << " += dim_mult * dimension_indices[" << dim_indices[i][j] << "];";
                        if(j > 0)
                            body << "dim_mult *= " << dim_sizes[dim_indices[i][j]] << ";\n";
                    }
                    body << "storage[" << std::get<uint32_t>(out_index) << "] = array" << i << ".data[" << input_index_var << "];\n";
                }
            }
            break;
        case op_codes::store:
            for(auto&& [out_index, i]: util::enumerate(output_indices)){
                if(!std::holds_alternative<size_t>(out_index))
                    continue;
                body << "vec array" << i << " = vec(" << std::get<size_t>(out_index) << "ul);\n";
                body << "array" << i << ".data[gl_GlobalInvocationID.x] = storage[" << std::get<uint32_t>(input_indices[i]) << "];\n";
            }
            break;
        case op_codes::pipeline_barrier:
            body << "}\n";
            header << "float storage[" << storage_size << "];" << body.str();
            pipelines.emplace_back(create_pipeline(header.str(), calc_thread_amt(data_state)));
            header.clear(); body.clear(); storage_size = 0;
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
            body << "storage[" << std::get<uint32_t>(output_indices[0]) << "] = float(gl_GlobalInvocationID.x);\n";
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
    if(storage_size){
        body << "}\n";
        header << "float storage[" << storage_size << "];" << body.str();
        pipelines.emplace_back(create_pipeline(header.str(), calc_thread_amt(data_state)));
    }
    return pipelines;
}
}