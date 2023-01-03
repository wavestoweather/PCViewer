#pragma once
#include <workbench_base.hpp>
#include <imgui.h>

namespace workbenches{
class drawlist_colors_workbench: public structures::workbench{
    struct session_state_t{
        std::string         color_palette_name{"Paired"}; // is either the name of a color_brewer palette or "custom"
        int                 color_palette_color_count{12};
        std::vector<ImColor> colors{};
        int                 cur_color{};
    } _session_state{};

public:
    drawlist_colors_workbench(std::string_view id);

    void show() override;

    ImU32 get_next_imu32();
    ImColor get_next_imcolor();

    // no settings, only session data can be restored
    void                set_session_data(const crude_json::value& json) override {}
    crude_json::value   get_session_data() const {return {};}
};
}

namespace globals{
const std::string_view drawlist_color_wb_id{"Drawlist color workbench"};
}