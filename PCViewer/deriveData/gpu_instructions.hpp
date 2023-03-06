#pragma once
#include <string_view>
#include <sstream>
#include <ranges.hpp>
#include "../imgui_nodes/crude_json.h"
#include "MemoryView.hpp"

namespace deriveData{
// intermediate representation of the calculation code has the following layout:
//      op_code [add1, ..., addn] [add_out_1, ..., add_out_n] ([dim_size1, ..., dim_sizen] [[add1 dim_1, add1 ... dim_n], ..., [addn_dim_1, ... addn_dim_n]]) {json with user defined data}
// for equal layouts/direct column operations the column size is given by dim_size_1 = -col.size
// the (...) part is optional and has to be checked for

// handling different data layouts by adding an optional division array which is used for dimension index calculation, followed by the dimension indices needed for index calculation

enum op_codes: uint32_t{
    none,                // used eg. by print vector nodes as they only download the data and print it
    load,
    store,
    pipeline_barrier,
    one_vec,
    zero_vec,
    rand_vec,
    iota_vec,
    copy,
    sum,
    product,
    lp_norm,
    inverse,
    negate,
    abs,
    square,
    sqrt,
    exp,
    log,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    ceil,
    floor,
    sinh,
    cosh,
    tanh,
    plus,
    minus,
    multiplication,
    division,
    pow,
    min,
    max,
    min_red,
    max_red,
    sum_red,
    mul_red,
    avg_red,
    stddev_red,
};
static std::ostream& operator<<(std::ostream& s, op_codes o){
    return s << static_cast<uint32_t>(o);
}
static std::istream& operator>>(std::istream& s, op_codes& o){
    return s >> reinterpret_cast<uint32_t&>(o);
}
//static std::stringstream& operator>>(std::stringstream& s, op_codes& o){
//    return s >> reinterpret_cast<uint32_t&>(o);
//}

using float_column_views = std::vector<column_memory_view<float>>;
struct add_flags{bool force_input_global_address:1; bool force_output_global_address:1;};
inline void add_operation(std::stringstream& operations, op_codes op_code, const float_column_views& inputs, const float_column_views& outputs, const crude_json::value& additional_data = {}, add_flags flags = {}){
    operations << op_code;
    if(inputs.empty() && outputs.empty()){
        operations << "\n";
        return;
    }
    // input addresses
    operations << " [";
    for(auto&& [in, last]: util::last_iter(inputs)){
        if(in.is_constant())
            operations << "c" << in.cols[0][0];
        else
            operations << "g" << in.cols[0].data();
        if(!last) operations << ",";
    }
    operations << "] [";
    // output addresses
    for(auto&& [in, last]: util::last_iter(outputs)){
        if(in.is_constant() && (op_code < op_codes::min_red || op_code > op_codes::stddev_red) && !flags.force_output_global_address)
            operations << "c" << in.cols[0][0];
        else
            operations << "g" << in.cols[0].data();
        if(!last) operations << ",";
    }
    operations << "] ";

    // dimension infos
    operations << "([";
    deriveData::memory_view<uint32_t> dimensions;
    for(size_t i: util::size_range(inputs))
        if(!inputs[i].dimensionSizes.empty()) dimensions = inputs[i].dimensionSizes;
    if(dimensions.empty())
        for(size_t i: util::size_range(outputs))
            if(!outputs[i].dimensionSizes.empty()) dimensions = outputs[i].dimensionSizes;
    
    for(auto&& [dim, last]: util::last_iter(dimensions)){
        operations << dim;
        if(!last) operations << ",";
    }
    operations << "] [";
    // per input dimensions
    for(auto&& [input, last]: util::last_iter(inputs)){
        operations << "[";
        for(auto&& [dim_index, last]: util::last_iter(input.columnDimensionIndices)){
            operations << dim_index;
            if(!last)
                operations << ",";
        }
        operations << "]";
        if(!last) operations << " ";
    }
    for(auto& output: outputs){
        operations << " [";
        for(auto&& [dim_index, last]: util::last_iter(output.columnDimensionIndices)){
            operations << dim_index;
            if(!last)
                operations << ",";
        }
        operations << "]";
    }
    operations << "])";

    if(!additional_data.is_null())
        operations << " {" << additional_data.dump() << '}';

    operations << '\n';
}
}