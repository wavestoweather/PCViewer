#include "drawlist_colors_workbench.hpp"
#include <color_brewer_util.hpp>
#include <util.hpp>

namespace workbenches{
drawlist_colors_workbench::drawlist_colors_workbench(std::string_view id): workbench(id){
    _session_state.colors = util::color_brewer::brew_imcol(_session_state.color_palette_name, _session_state.color_palette_color_count);
}

void drawlist_colors_workbench::show(){
    if(!active)
        return;
    
    ImGui::Begin(id.data(), &active, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking);

    for(auto&& [c, i]: util::enumerate(_session_state.colors)){
        ImGui::PushID(i);
        if(i != 0)
            ImGui::SameLine();
        ImGui::ColorEdit4("##dl_cols", &c.Value.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_DisplayRGB);
        if(i == _session_state.cur_color){
            ImVec2 min = ImGui::GetItemRectMin();
            ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRect(min, max, IM_COL32(80, 80, 255, 255), 0, 0, 2);
        }
        ImGui::PopID();
    }
    const auto& palette_infos = brew_palette_infos();
    if(ImGui::BeginCombo("Select palette", _session_state.color_palette_name.c_str())){
        for(const auto& info: palette_infos){
            if(ImGui::MenuItem(info.name.data())){
                _session_state.color_palette_name = info.name;
                _session_state.cur_color = 0;
                _session_state.color_palette_color_count = std::clamp(_session_state.color_palette_color_count, info.min_colors, info.max_colors);
                _session_state.colors = util::color_brewer::brew_imcol(_session_state.color_palette_name, _session_state.color_palette_color_count);
            }
        }
        ImGui::EndCombo();
    }
    const auto& palette_info = *std::find_if(palette_infos.begin(), palette_infos.end(), [&](const palette_info_t& i){return i.name == _session_state.color_palette_name;});
    if(ImGui::BeginCombo("Set color count", std::to_string(_session_state.color_palette_color_count).c_str())){
        for(int i: util::i_range(palette_info.min_colors, palette_info.max_colors + 1)){
            if(ImGui::MenuItem(std::to_string(i).c_str())){
                _session_state.color_palette_color_count = i;
                _session_state.colors = util::color_brewer::brew_imcol(_session_state.color_palette_name, _session_state.color_palette_color_count);
            }
        }
        ImGui::EndCombo();
    }

    ImGui::End();
}

ImU32 drawlist_colors_workbench::get_next_imu32(){
    if(_session_state.colors.size() == 0)
        return IM_COL32_WHITE;
    
    ImU32 c = _session_state.colors[_session_state.cur_color++];
    _session_state.cur_color %= _session_state.colors.size();
    return c;
}

ImColor drawlist_colors_workbench::get_next_imcolor(){
    if(_session_state.colors.size() == 0)
        return {};

    ImColor c = _session_state.colors[_session_state.cur_color++];
    _session_state.cur_color %= _session_state.colors.size();
    return c;
}
}