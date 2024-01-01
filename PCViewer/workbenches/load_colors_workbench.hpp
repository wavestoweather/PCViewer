#pragma once
#include <workbench_base.hpp>
#include <imgui.h>

namespace workbenches{
class load_colors_workbench: public structures::workbench{
    struct session_state_t{
        std::string         drawlist_color_palette_name{"Set3"}; // is either the name of a color_brewer palette or "custom"
        std::string         drawlist_color_palett_type{"Qualitative"};
        int                 drawlist_color_palette_color_count{12};
        std::vector<ImColor> drawlist_colors{};
        int                 drawlist_cur_color{};

        std::string         attribute_color_palette_name{"Set3"}; // is either the name of a color_brewer palette or "custom"
        std::string         attribute_color_palett_type{"Qualitative"};
        int                 attribute_color_palette_color_count{12};
        std::vector<ImColor> attribute_colors{};
        int                 attribute_cur_color{};
    } _session_state{};

    void _imgui_color_selection(std::string& color_pallette_name, std::string& color_palette_type, int& color_palette_color_count, std::vector<ImColor>& colors, int& cur_color);
public:
    load_colors_workbench(std::string_view id);

    void show() override;

    ImU32 get_next_drawlist_imu32();
    ImColor get_next_drawlist_imcolor();
    ImU32 get_next_attribute_imu32();
    ImColor get_next_attribute_imcolor();

    // no settings, only session data can be restored
    void                set_session_data(const crude_json::value& json) override {}
    crude_json::value   get_session_data() const {return {};}
};
}

namespace globals{
const std::string_view load_color_wb_id{"Load color workbench"};
}