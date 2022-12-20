#pragma once
#include <enum_names.hpp>

namespace structures{
enum class data_type_t: uint32_t{
    float_t,
    half_t,
    uint_t,
    ushort_t,
    COUNT
};
const structures::enum_names<data_type_t> data_type_names{
    "float",
    "half",
    "uint",
    "ushort"
};
}