#pragma once
#include <c_file.hpp>
#include <filesystem>
#include <fstream>
#include "../imgui_nodes/crude_json.h"

namespace util{
namespace json{
inline crude_json::value open_json(std::string_view filename){
    structures::c_file file(filename, "rb");
    if(!file)
        throw std::runtime_error{"util::json::open_json() File " + std::string(filename) + " could not be found."};
    std::string json_string; json_string.resize(std::filesystem::file_size(std::string(filename)));
    file.read<char>({json_string.data(), json_string.size()});
    return crude_json::value::parse(json_string);
}

// for indent < 0 dumps json as single string,
// for indent == 0 dumps json line by line without indent
// fur indent >0 dumps json line by line and indents inner contents by indent spaces
inline void save_json(std::string_view filename, const crude_json::value& json, int indent = -1){
    std::ofstream file(std::string(filename), std::ios::binary);
    file << json.dump(indent);
}
}
}