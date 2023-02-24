#pragma once
#include <shader_compiler.hpp>
#include <vk_util.hpp>
#include <logger.hpp>
#include <sstream>
#include <charconv>
#include <robin_hood.h>
#include <variant>
#include <fast_float.h>
#include <ranges.hpp>
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
        std::from_chars(address.data() + 1,  address.data() + address.size(), std::get<uint32_t>(ret));
        break;
    case 'g':
        ret = size_t{};
        std::from_chars(address.data() + 1,  address.data() + address.size(), std::get<size_t>(ret));
        break;
    case 'c':
        ret = float{};
        fast_float::from_chars(address.data() + 1,  address.data() + address.size(), std::get<float>(ret));
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
        std::from_chars(size.data(), size.data() + size.size(), dimension_sizes.back());
    }
    dimension_info = dimension_info.substr(0, dimension_info.find('{'));
    auto tmp = dimension_info | util::slice('[');
    for(auto dimension_indices: util::subrange(tmp.begin() + 3, tmp.end())){
        dimension_indices_v.emplace_back();
        auto dim_inds = (dimension_indices | util::slice(']'))[0];
        for(auto ind: dim_inds | util::slice(',')){
            dimension_indices_v.back().emplace_back();
            std::from_chars(ind.data(), ind.data() + ind.size(), dimension_indices_v.back().back());
        }
    }
    return {std::move(dimension_sizes), std::move(dimension_indices_v)};
}
inline std::stringstream addresses_to_stringstream(const std::vector<data_storage>& addresses){
    std::stringstream adds; adds << '[';
    for(auto&& [in, last]: util::last_iter(addresses)){
        if(std::holds_alternative<uint64_t>(in)) adds << 'g' << std::get<uint64_t>(in);
        if(std::holds_alternative<uint32_t>(in)) adds << 'l' << std::get<uint32_t>(in);
        if(std::holds_alternative<float>(in)) adds << 'c' << std::get<float>(in);
        if(!last) adds << ',';
    }
    adds << ']';
    return adds;
}
inline std::stringstream get_add_local_addresses(std::vector<std::variant<uint64_t, float>>& local_addresses, const std::vector<data_storage>& indices){
    std::stringstream local_adds; local_adds << '[';
    for(auto&& [in, last]: util::last_iter(indices)){
        size_t storage_pos{util::n_pos};
        if(std::holds_alternative<uint64_t>(in)) storage_pos = local_addresses | util::index_of(std::variant<uint64_t, float>(std::get<uint64_t>(in))); 
        if(storage_pos == util::n_pos && std::holds_alternative<uint64_t>(in))
            local_addresses.emplace_back(std::get<uint64_t>(in));
        else if(storage_pos == util::n_pos)
            local_addresses.emplace_back(std::get<float>(in));
        if(storage_pos == util::n_pos) local_adds << 'l' << local_addresses.size() - 1;
        else local_adds << 'l' << storage_pos;
        if(!last) local_adds << ',';
    }
    local_adds << ']';
    return local_adds;
}
inline void flush_locals(const std::vector<std::variant<uint64_t, float>>& local_storage_addresses, const std::vector<uint32_t>& dim_sizes, const std::vector<std::vector<uint32_t>>& local_storage_dim_indices, std::stringstream& result_stream){
    std::stringstream storage_out_pos, internal_out_pos;
    storage_out_pos << '['; internal_out_pos << "[";
    for(auto&& [add, i]: util::enumerate(local_storage_addresses)){
        if(std::holds_alternative<float>(add)) continue;
        if(storage_out_pos.str().size() > 1) {storage_out_pos << ','; internal_out_pos << ',';}
        storage_out_pos << 'g' << std::get<uint64_t>(add);
        internal_out_pos << 'l' << i;
    }
    storage_out_pos << ']'; internal_out_pos << ']';
    result_stream << op_codes::store << ' ' << internal_out_pos.str() << ' ' << storage_out_pos.str() << " ([";
    for(auto&& [dim_size, last]: util::last_iter(dim_sizes)){
        result_stream << dim_size;
        if(!last) result_stream << ',';
    }
    result_stream << "] [";
    bool first_print{};
    for(const auto& dim_indices: local_storage_dim_indices){
        if(dim_indices.empty()) continue;
        if(!first_print) result_stream << ' ';
        result_stream << '[';
        for(auto&& [dim_ind, last]: util::last_iter(dim_indices)){
            result_stream << dim_ind;
            if(!last) result_stream << ',';
        }
        result_stream << ']';
        first_print = false;
    }
    result_stream << "])\n";

    result_stream << op_codes::pipeline_barrier << " [] [] ([])\n";
}

// checks if all input/output indices are the same, ignores constants (indices which are empty)
inline bool equal_data_layout(const std::vector<std::vector<uint32_t>>& input_output_indices){
    size_t ref{};
    for(size_t i: util::i_range(size_t(1), input_output_indices.size())){
        if(input_output_indices[i].empty()) continue;
        if(input_output_indices[ref].empty()) ref = i;
        else{
            if(input_output_indices[ref].size() != input_output_indices[i].size()) return false;
            for(size_t j: util::size_range(input_output_indices[i]))
                if(input_output_indices[ref][j] != input_output_indices[i][j]) return false;
        }
    }
    return true;
}

// adds all the load store operations in a possibly optimal manner
// this also means, that the mapping from global to local address space is done here
// an example would be:
// op_vec_iota a b
// ->
// op_load a 0
// op_vec_iota 0
// op_store 0 b
// where a and b are global addresses, and 0 is the index in the local storage
inline std::string optimize_operations(const std::string& input){
    std::stringstream result_stream;
    std::vector<std::variant<uint64_t, float>> local_storage_addresses;                  // 0 indicates constant address which has no 
    std::vector<std::vector<uint32_t>> local_storage_dim_indices;
    std::vector<uint32_t> cur_dim_sizes;
    std::vector<uint32_t> cur_dim_indices;
    std::vector<uint32_t> merged_dims_storage;
    std::string_view rest;
    for(std::string_view line: input | util::slice('\n')){
        auto slice = (line | util::slice(' ')).begin();
        std::string_view op_code = *slice++;
        std::string_view input = *slice++, output = *slice, rest = slice.get_rest();
        const auto [dim_sizes, dim_indices] = extract_input_output_dimensions(line);
        const auto [input_indices, output_indices] = extract_input_output_indices(line);
        const bool equal_dim_indices = equal_data_layout(dim_indices) && dim_indices[input_indices.size()].size();
        for(const auto& dim_index: dim_indices){
            if(dim_index.empty()) continue;
            if(merged_dims_storage.empty()) merged_dims_storage = dim_index;
            else if(merged_dims_storage != dim_index){merged_dims_storage = util::size_range(dim_sizes) | util::to<std::vector<uint32_t>>();}
        }
        if(cur_dim_sizes.size() && (cur_dim_sizes != dim_sizes || cur_dim_indices != merged_dims_storage)){
            // storing
            flush_locals(local_storage_addresses, cur_dim_sizes, local_storage_dim_indices, result_stream);
            local_storage_addresses.clear(); cur_dim_sizes.clear(); cur_dim_indices.clear();
        }
        cur_dim_sizes = dim_sizes;
        cur_dim_indices = merged_dims_storage;

        size_t prev_storage_size = local_storage_addresses.size();
        auto input_local_addresses = get_add_local_addresses(local_storage_addresses, input_indices);
        if(prev_storage_size < local_storage_addresses.size()){
            // load missing values
            result_stream << op_codes::load << " [";
            for(size_t i: util::i_range(prev_storage_size, local_storage_addresses.size())){
                if(std::holds_alternative<float>(local_storage_addresses[i])) result_stream << 'c' << std::get<float>(local_storage_addresses[i]);
                else result_stream << 'g' << std::get<uint64_t>(local_storage_addresses[i]);
            }
            result_stream << "] [";
            for(size_t i: util::i_range(prev_storage_size, local_storage_addresses.size())){
                result_stream << 'l' << i;
                if(i != local_storage_addresses.size() -  1) result_stream << ",";
            }
            result_stream << "] " << rest << '\n';
        }
        std::stringstream output_local_addresses;
        if(equal_dim_indices){
            output_local_addresses = get_add_local_addresses(local_storage_addresses, output_indices);
            if(local_storage_addresses.size() != local_storage_dim_indices.size()){
                for(size_t i: util::i_range(local_storage_dim_indices.size(), local_storage_addresses.size())){
                    local_storage_dim_indices.emplace_back();
                    if(std::holds_alternative<float>(local_storage_addresses[i])) continue;
                    size_t ind = input_indices | util::index_of(data_storage(std::get<uint64_t>(local_storage_addresses[i])));
                    if(ind == util::n_pos)
                        ind = input_indices.size() + (output_indices | util::index_of(data_storage(std::get<uint64_t>(local_storage_addresses[i]))));
                    assert(ind != util::n_pos);
                    local_storage_dim_indices.back() = dim_indices[ind];
                }
            }
        }
        else{
            output_local_addresses = addresses_to_stringstream(output_indices);
            for(const auto& input: input_indices){
                if(std::holds_alternative<float>(input) || std::holds_alternative<uint32_t>(input)) continue;
                auto storage_address = local_storage_addresses | util::try_find(std::variant<uint64_t, float>{std::get<uint64_t>(input)});
                storage_address->get() = .0f;
            }
        }

        // main operation
        result_stream << op_code << ' ' << input_local_addresses.str() << ' ' << output_local_addresses.str() << ' ' << rest << '\n';
    }
    if(cur_dim_sizes.size())
        flush_locals(local_storage_addresses, cur_dim_sizes, local_storage_dim_indices, result_stream);

    if(logger.logging_level >= logging::level::l_5)
        logger << logging::info_prefix << ' ' << result_stream.str();
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
        if(logger.logging_level >= logging::level::l_5){
            std::string_view c = std::string_view(code).substr(code.find('\n') + 1);
            std::stringstream formatted_code;
            int cur_indent{};
            int line_number{};
            for(std::string_view line: c | util::slice('\n')){
                size_t trim_pos =  line.find_first_not_of(" ");
                if(trim_pos != std::string_view::npos) line = line.substr(trim_pos);
                formatted_code << ++line_number << ':';
                cur_indent -= int(std::count(line.begin(), line.end(), '}'));
                cur_indent = std::max(cur_indent, 0);
                for(int i: util::i_range(cur_indent * 4)) formatted_code << ' ';
                formatted_code << line << '\n';
                cur_indent += int(std::count(line.begin(), line.end(), '{'));
            }
            logger << "deriveData()::create_gpu_pipelines() New pipeline created with code:\n" << formatted_code.str() << logging::endl;
        }
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
    std::stringstream           body;
    robin_hood::unordered_set<uint32_t> declared_locals;
    for(std::string_view line: instructions | util::slice('\n')){
        std::stringstream line_stream{std::string(line.substr(0, line.find(' ')))};
        op_codes operation;
        line_stream >> operation;
        const auto [dim_sizes, dim_indices] = extract_input_output_dimensions(line);
        const auto [input_indices, output_indices] = extract_input_output_indices(line);
        // check dimension consistency
        for(const auto& i: dim_indices){
            for(uint32_t dim: i){
                if(!(data_state.cur_dimension_indices | util::contains(dim)))
                    data_state.cur_dimension_indices.emplace_back(dim);
            }
        }
        const bool same_before = data_state.all_same_data_layout;
        if(dim_indices.size())
            for(size_t i: util::i_range(dim_indices.size() - 1))
                data_state.all_same_data_layout &= dim_indices[i] == dim_indices[i + 1] || dim_indices[i].empty() || dim_indices[i + 1].empty();
        assert(!same_before || same_before == data_state.all_same_data_layout);
        assert(operation == op_codes::pipeline_barrier || data_state.cur_dimension_sizes.empty() || dim_sizes.empty() || data_state.cur_dimension_sizes == dim_sizes);
        if(operation != op_codes::pipeline_barrier)
            data_state.cur_dimension_sizes = dim_sizes;

        // header
        if(body.str().empty()){
            body << R"(
                #version 450
                #extension GL_EXT_buffer_reference2: require
                #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
                #extension GL_EXT_scalar_block_layout: require
                #extension GL_EXT_shader_atomic_float: require

                layout(buffer_reference, scalar) buffer vec{
                    float data[];
                };

                layout(push_constant) uniform PCs{
                    uint rand_seed;
                };

                layout(local_size_x = )" STR_IND(workgroup_size) R"() in;

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

        crude_json::value additional_data{}; auto start = line.find('{');
        if(start != std::string_view::npos){
            auto end = line.find_last_of("}");
            additional_data = crude_json::value::parse(std::string(line.substr(start, end - start - 1)));
        }

        // locals declaration
        for(auto input: output_indices){
            if(std::holds_alternative<uint32_t>(input) && !declared_locals.contains(std::get<uint32_t>(input))){
                body << "float storage" << std::get<uint32_t>(input) << ";\n";
                declared_locals.insert(std::get<uint32_t>(input));
            }
        }
        // instruction decoding
        switch(operation){
        case op_codes::none:
            break;
        case op_codes::load:

            for(auto&& [input_index, i]: util::enumerate(input_indices))
                if(std::holds_alternative<size_t>(input_index))
                    body << "vec array"<< i <<" = vec(" << std::get<size_t>(input_index) << "ul);\n";
            if(data_state.all_same_data_layout){
                for(auto&& [out_index, i]: util::enumerate(output_indices)){
                    if(std::holds_alternative<float>(input_indices[i]))
                        body << "storage" << std::get<uint32_t>(out_index) << " = " << std::get<float>(input_indices[i]) << ";\n";
                    else
                        body << "storage" << std::get<uint32_t>(out_index) << " = array" << i << ".data[gl_GlobalInvocationID.x];\n";
                }
            }
            else{
                body << "float dimension_indices["<< data_state.cur_dimension_sizes.size() <<"]; uint i = gl_GlobalInvocationID.x;\n";
                for(size_t i: util::rev_size_range(data_state.cur_dimension_sizes))
                    body << "dimension_indices[" << i << "] = i % " << data_state.cur_dimension_sizes[i] << "; i /= " << data_state.cur_dimension_sizes[i] << ";\n";
                body << "uint dim_mult;\n";
                for(auto&& [out_index, i]: util::enumerate(output_indices)){
                    if(std::holds_alternative<float>(input_indices[i])){
                        body << "storage" << std::get<uint32_t>(out_index) << " = " << std::get<float>(input_indices[i]) << ";\n";
                        continue;
                    }
                    std::string input_index_var("in" + std::to_string(i));
                    body << "uint " << input_index_var << " = 0; dim_mult = 1;\n";
                    for(size_t j: util::rev_size_range(dim_indices[i])){
                        body << input_index_var << " += dim_mult * dimension_indices[" << dim_indices[i][j] << "];";
                        if(j > 0)
                            body << "dim_mult *= " << dim_sizes[dim_indices[i][j]] << ";\n";
                    }
                    body << "storage" << std::get<uint32_t>(out_index) << " = array" << i << ".data[" << input_index_var << "];\n";
                }
            }
            break;
        case op_codes::store:
            for(auto&& [out_index, i]: util::enumerate(output_indices)){
                if(!std::holds_alternative<size_t>(out_index))
                    continue;
                body << "vec array_out" << i << " = vec(" << std::get<size_t>(out_index) << "ul);\n";
                body << "array_out" << i << ".data[gl_GlobalInvocationID.x] = storage" << std::get<uint32_t>(input_indices[i]) << ";\n";
            }
            break;
        case op_codes::pipeline_barrier:
            body << "}\n";
            pipelines.emplace_back(create_pipeline(body.str(), calc_thread_amt(data_state)));
            body.str({}); declared_locals.clear(); data_state = {};
            break;
        case op_codes::one_vec:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = 1;\n";
            break;
        case op_codes::zero_vec:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = 0;\n";
            break;
        case op_codes::rand_vec:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = random_float();\n";
            break;
        case op_codes::iota_vec:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = float(gl_GlobalInvocationID.x);\n";
            break;
        case op_codes::copy:
            if(output_indices[0] != input_indices[0] && dim_sizes.size())
                body << "storage" << std::get<uint32_t>(output_indices[0]) << " = storage" << std::get<uint32_t>(input_indices[0]) << ";\n";
            break;
        case op_codes::sum:
            // TODO additional data, such as pre factors
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = ";
            for(auto&& [e, last]: util::last_iter(input_indices)){
                body << "storage" << std::get<uint32_t>(e);
                if(!last) body << " + ";
            }
            body << ";\n";
            break;
        case op_codes::product:
            // TODO additional data, such as pre factors
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = ";
            for(auto&& [e, last]: util::last_iter(input_indices)){
                body << "storage" << std::get<uint32_t>(e);
                if(!last) body << " * ";
            }
            body << ";\n";
            break;
        case op_codes::lp_norm:
            // TODO: get norm from additional data
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = sqrt(";
            for(auto&& [e, last]: util::last_iter(input_indices)){
                body << "storage" << std::get<uint32_t>(e) << " * storage" << std::get<uint32_t>(e);
                if(!last) body << " + ";
            }
            body << ");\n";
            break;
        case op_codes::inverse:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = 1. / storage" << std::get<uint32_t>(input_indices[0]) << ";\n";
            break;
        case op_codes::negate:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = -storage" << std::get<uint32_t>(input_indices[0]) << ";\n";
            break;
        case op_codes::abs:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = abs(storage" << std::get<uint32_t>(input_indices[0]) << ");\n";
            break;
        case op_codes::square:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = storage" << std::get<uint32_t>(input_indices[0]) << " * storage" << std::get<uint32_t>(input_indices[0]) << ";\n";
            break;
        case op_codes::sqrt:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = sqrt(storage" << std::get<uint32_t>(input_indices[0]) << ");\n";
            break;
        case op_codes::exp:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = exp(storage" << std::get<uint32_t>(input_indices[0]) << ");\n";
            break;
        case op_codes::log:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = log(storage" << std::get<uint32_t>(input_indices[0]) << ");\n";
            break;
        case op_codes::plus:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = storage" << std::get<uint32_t>(input_indices[0]) << " + storage" << std::get<uint32_t>(input_indices[1]) << ";\n";
            break;
        case op_codes::minus:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = storage" << std::get<uint32_t>(input_indices[0]) << " - storage" << std::get<uint32_t>(input_indices[1]) << ";\n";
            break;
        case op_codes::multiplication:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = storage" << std::get<uint32_t>(input_indices[0]) << " * storage" << std::get<uint32_t>(input_indices[1]) << ";\n";
            break;
        case op_codes::division:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = storage" << std::get<uint32_t>(input_indices[0]) << " / storage" << std::get<uint32_t>(input_indices[1]) << ";\n";
            break;
        case op_codes::pow:
            body << "storage" << std::get<uint32_t>(output_indices[0]) << " = pow(storage" << std::get<uint32_t>(input_indices[0]) << ", storage" << std::get<uint32_t>(input_indices[1]) << ");\n";
            break;
        case op_codes::sum_red:
            body << "vec array_out" << std::get<uint32_t>(input_indices[0]) << " = vec(" << std::get<size_t>(output_indices[0]) << "ul);\n";
            body << "atomicAdd(array_out" << std::get<uint32_t>(input_indices[0]) << ".data[uint(storage" << std::get<uint32_t>(input_indices[0]) <<")], storage" << std::get<uint32_t>(input_indices[1]) << ");\n";
            break;
        default:
            throw std::runtime_error{"Unsupported op_code for operation line: " + std::string(line) + ". Use Cpu execution for this execution graph."};
        }
    }
    if(declared_locals.size()){
        body << "}\n";
        pipelines.emplace_back(create_pipeline(body.str(), calc_thread_amt(data_state)));
    }
    return pipelines;
}
}