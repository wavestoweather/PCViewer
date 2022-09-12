#pragma once
#include <vector>
#include <string_view>
#include <c_file.hpp>

namespace util{
inline std::vector<uint32_t> read_file(std::string_view filename){
    structures::c_file input(filename, "rb");
    std::vector<uint32_t> data;
    input.read(util::memory_view(data));
    return data;
}
}