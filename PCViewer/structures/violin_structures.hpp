#pragma once
#include <drawlists.hpp>
#include <workbench_base.hpp>
#include <std_util.hpp>

namespace structures{
namespace violins{
using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
using attribute_order_info = structures::attribute_order_info;

struct drawlist_attribute{
    std::string_view dl;
    int              attribute;

    bool operator==(const drawlist_attribute& da) const {return dl == da.dl && attribute == da.attribute;}

    DECL_DRAWLIST_READ(dl)
};

struct histogram{
    std::vector<float>  smoothed_values; // the original values of the histogram can be found in the histogram manager of the corresponding drawlist
    float               area{};
    float               max_val{};
};

struct dl_violin_appearance{
    float   fill_alpha{.3f};
    float   line_alpha{1.f};
    bool    decouple_line_color{};
    ImVec4  line_color{.0f,.0f,.0f,1.f};
};

struct drawlist_info{
    std::string_view                        drawlist_id{};
    bool                                    linked_with_drawlist{true};
    util::memory_view<appearance_tracker>   appearance{};
    dl_violin_appearance                    violin_appearance{};

    DECL_DRAWLIST_READ(drawlist_id)
    DECL_DRAWLIST_WRITE(drawlist_id)
    DECL_DATASET_READ(drawlist_read().parent_dataset)
    DECL_DATASET_WRITE(drawlist_read().parent_dataset)
    DECL_DL_TEMPLATELIST_READ(drawlist_id)
};

struct session_common{
    std::vector<drawlist_info>  drawlist_infos{};
    std::vector<attribute>      attributes{};
    std::vector<attribute_order_info> attribute_order_infos{};
    std::vector<uint8_t>        attribute_log{};
};
struct drawlist_session_state_t: public session_common{
    std::array<int, 2> matrix_dimensions;
};
struct attribute_session_state_t: public session_common{
};

struct settings_common{
    ImVec4  plot_backgroun{.0f, .0f, .0f, 1.f};
    float   line_hover_dist{5.f};
    int     histogram_bin_count{100};
    float   smoothing_std_dev{-1};
    bool    ignore_zero_bins{false};
};
struct attribute_settings_t: public settings_common{

};
struct drawlist_settings_t: public settings_common{
    std::array<int, 2> matrix_dimensions{5, 2};
};
}
}

template<> struct std::hash<structures::violins::drawlist_attribute>{
    inline size_t operator()(const structures::violins::drawlist_attribute& da) const{
        size_t hash = std::hash<std::string_view>{}(da.dl);
        return std::hash_combine(hash, da.attribute);
    }
};