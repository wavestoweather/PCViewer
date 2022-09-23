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

// returns in the first string_view the filename, in the second the extension (including the .)
inline std::tuple<std::string_view, std::string_view> get_file_extension(std::string_view filename){
    std::string_view file = filename.substr(filename.find_last_of("/\\") + 1);
    size_t extension_pos = file.find_last_of(".");
	std::string_view file_extension = extension_pos == std::string_view::npos ? std::string_view{""}: file.substr(extension_pos);
    return {file, file_extension};
}
}