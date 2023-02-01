#pragma once
#include <string_view>
#include <sstream>
#include <ranges.hpp>
#include "MemoryView.hpp"

namespace deriveData{
// intermediate representation of the calculation code has the following layout:
//      op_code [add1, ..., addn] [add_out_1, ..., add_out_n] ([dim_size1, ..., dim_sizen] [[add1 dim_1, add1 ... dim_n], ..., [addn_dim_1, ... addn_dim_n]])
// for equal layouts/direct column operations the column size is given by dim_size_1 = -col.size
// the (...) part is optional and has to be checked for

// handling different data layouts by adding an optional divison array which is used for dimension index calculation, followed by the dimension indices needed for index calculation

enum op_codes: uint32_t{
    none,                // used eg. by print vector nodes as they only download the data and print it
    pipeline_barrier,
    one_vec,
    zero_vec,
    rand_vec,
    iota_vec,
    copy,
    sum,
};
static std::ostream& operator<<(std::ostream& s, op_codes o){
    return s << static_cast<uint32_t>(o);
}
static std::istream& operator>>(std::istream& s, op_codes& o){
    return s >> reinterpret_cast<uint32_t&>(o);
}

using float_column_views = std::vector<column_memory_view<float>>;
inline void add_operation(std::stringstream& operations, op_codes op_code, const float_column_views& inputs, const float_column_views& outputs){
    operations << op_code;
    if(inputs.empty() && outputs.empty()){
        operations << "\n";
        return;
    }
    // input addresses
    operations << " [";
    for(auto&& [in, last]: util::last_iter(inputs)){
        operations << in.cols[0].data();
        if(!last) operations << ",";
    }
    operations << "] [";
    // output addresses
    for(auto&& [in, last]: util::last_iter(outputs)){
        operations << in.cols[0].data();
        if(!last) operations << ",";
    }
    operations << "] ";
    bool same_layout{true};
    if(inputs.size() > 1)
        for(size_t i: util::i_range(inputs.size() - 1))
            same_layout &= inputs[i].equalDataLayout(inputs[i + 1]);

    if(same_layout){
        operations << "\n";
        return;
    }

    // dimension infos
    operations << "([";
    for(auto&& [dim, last]: util::last_iter(inputs[0].dimensionSizes)){
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
    operations << "])\n";
}
}