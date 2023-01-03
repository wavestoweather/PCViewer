#pragma once
#include <colorbrewer.h>
#include <imgui.h>
#include <ranges.hpp>
#include <sstream>

namespace util{
namespace color_brewer{
inline std::vector<ImU32> brew_u32(const std::string& color_name, size_t color_count){
    auto colors = brew<std::string_view>(color_name, color_count);

    std::stringstream converter; converter << std::hex; // setting the converter to hex
    std::vector<ImU32> output(colors.size());
    for(const auto&& [v, i]: util::indexed_iter(colors)){
        converter << v.substr(1);
        converter >> output[i];
        output[i] |= 255 << IM_COL32_A_SHIFT;
    }
    return output;
}

inline std::vector<ImColor> brew_imcol(const std::string& color_name, size_t color_count){
    auto colors = brew<std::string_view>(color_name, color_count);

    ImU32 c;
    std::vector<ImColor> output(colors.size());
    for(const auto&& [v, i]: util::indexed_iter(colors)){
        std::stringstream converter; 
        converter << std::hex << v.substr(1);
        converter >> c;
        output[i] = ImColor(c | (255 << IM_COL32_A_SHIFT));
        output[i].Value.w = 1;
        std::swap(output[i].Value.x, output[i].Value.z);
    }
    return output;
}
}
}