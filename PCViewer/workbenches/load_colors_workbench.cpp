#include "load_colors_workbench.hpp"
#include <color_brewer_util.hpp>
#include <util.hpp>
#include <imgui_util.hpp>
#include <ranges.hpp>

namespace workbenches{
void load_colors_workbench::_imgui_color_selection(std::string& color_palette_name, std::string& color_palette_type, int& color_palette_color_count, std::vector<ImColor>& colors, int& cur_color){
    for(auto&& [c, i]: util::enumerate(colors)){
        util::imgui::scoped_id col_id(static_cast<int>(i));
        if(i != 0)
            ImGui::SameLine();
        ImGui::ColorEdit4("##dl_cols", &c.Value.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_DisplayRGB);
        if(i == cur_color){
            ImVec2 min = ImGui::GetItemRectMin();
            ImVec2 max = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRect(min, max, IM_COL32(80, 80, 255, 255), 0, 0, 2);
        }
    }
    const auto& palette_infos = brew_palette_infos();

    if(ImGui::BeginCombo("Palette type", color_palette_type.c_str())){
        for(const auto& [type, palettes]: brew_palette_types())
            if(ImGui::MenuItem(type.data()))
                color_palette_type = type;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if(ImGui::BeginCombo("Palette color", color_palette_name.c_str())){
        for(const auto& palette: brew_palette_types().at(color_palette_type)){
            if(ImGui::MenuItem(palette.data())){
                const auto& palette_info = (palette_infos | util::try_find_if<const palette_info_t>([&](auto i){return i.name == palette;}))->get();
                color_palette_name = palette;
                cur_color = 0;
                color_palette_color_count = std::clamp(color_palette_color_count, palette_info.min_colors, palette_info.max_colors);
                colors = util::color_brewer::brew_imcol(color_palette_name, color_palette_color_count);
            }
            if(ImGui::IsItemHovered()){ // displaying a preview of the color map
                ImGui::BeginTooltip();
                const auto& palette_info = (palette_infos | util::try_find_if<const palette_info_t>([&](auto i){return i.name == palette;}))->get();
                for(const auto [color, first]: util::first_iter(util::color_brewer::brew_imcol(palette, palette_info.max_colors))){
                    if(!first) ImGui::SameLine();
                    ImGui::ColorButton("##c", color, 0, {ImGui::GetTextLineHeight(), ImGui::GetTextLineHeight()});
                }
                ImGui::EndTooltip();
            }
        }
        ImGui::EndCombo();
    }
    const auto& palette_info = *std::find_if(palette_infos.begin(), palette_infos.end(), [&](const palette_info_t& i){return i.name == color_palette_name;});
    if(ImGui::BeginCombo("Set color count", std::to_string(color_palette_color_count).c_str())){
        for(int i: util::i_range(palette_info.min_colors, palette_info.max_colors + 1)){
            if(ImGui::MenuItem(std::to_string(i).c_str())){
                color_palette_color_count = i;
                colors = util::color_brewer::brew_imcol(color_palette_name, color_palette_color_count);
            }
        }
        ImGui::EndCombo();
    }
}

load_colors_workbench::load_colors_workbench(std::string_view id): workbench(id)
{
    _session_state.drawlist_colors = util::color_brewer::brew_imcol(_session_state.drawlist_color_palette_name, _session_state.drawlist_color_palette_color_count);
    _session_state.attribute_colors = util::color_brewer::brew_imcol(_session_state.attribute_color_palette_name, _session_state.attribute_color_palette_color_count);
}

void load_colors_workbench::show(){
    if(!active)
        return;
    
    ImGui::Begin(id.data(), &active, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking);

    ImGui::Text("Drawlist colors:");
    ImGui::PushID("dl");
    _imgui_color_selection(_session_state.drawlist_color_palette_name, _session_state.drawlist_color_palett_type, _session_state.drawlist_color_palette_color_count, _session_state.drawlist_colors, _session_state.drawlist_cur_color);
    ImGui::PopID();
    ImGui::Separator();
    ImGui::PushID("at");
    ImGui::Text("Attribute colors:");
    _imgui_color_selection(_session_state.attribute_color_palette_name, _session_state.attribute_color_palett_type, _session_state.attribute_color_palette_color_count, _session_state.attribute_colors, _session_state.attribute_cur_color);
    ImGui::PopID();

    ImGui::End();
}

ImU32 load_colors_workbench::get_next_drawlist_imu32(){
    if(_session_state.drawlist_colors.size() == 0)
        return IM_COL32_WHITE;
    
    ImU32 c = _session_state.drawlist_colors[_session_state.drawlist_cur_color++];
    _session_state.drawlist_cur_color %= _session_state.drawlist_colors.size();
    return c;
}

ImColor load_colors_workbench::get_next_drawlist_imcolor(){
    if(_session_state.drawlist_colors.size() == 0)
        return {};

    ImColor c = _session_state.drawlist_colors[_session_state.drawlist_cur_color++];
    _session_state.drawlist_cur_color %= _session_state.drawlist_colors.size();
    return c;
}

ImU32 load_colors_workbench::get_next_attribute_imu32(){
    if(_session_state.drawlist_colors.size() == 0)
        return IM_COL32_WHITE;
    
    ImU32 c = _session_state.attribute_colors[_session_state.attribute_cur_color++];
    _session_state.attribute_cur_color %= _session_state.attribute_colors.size();
    return c;
}

ImColor load_colors_workbench::get_next_attribute_imcolor(){
    if(_session_state.attribute_colors.size() == 0)
        return {};

    ImColor c = _session_state.attribute_colors[_session_state.attribute_cur_color++];
    _session_state.attribute_cur_color %= _session_state.attribute_colors.size();
    return c;
}
}