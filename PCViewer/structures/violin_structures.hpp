#pragma once
#include <drawlists.hpp>
#include <workbench_base.hpp>
#include <std_util.hpp>

namespace structures{
namespace violins{
using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
using attribute_order_info = structures::attribute_info;

struct drawlist_attribute{
    std::string_view dl;
    std::string_view att;

    bool operator==(const drawlist_attribute& da) const {return dl == da.dl && att == da.att;}

    DECL_DRAWLIST_READ(dl);
    DECL_DATASET_READ(drawlist_read().parent_dataset);
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

    DECL_DRAWLIST_READ(drawlist_id)
    DECL_DRAWLIST_WRITE(drawlist_id)
    DECL_DATASET_READ(drawlist_read().parent_dataset)
    DECL_DATASET_WRITE(drawlist_read().parent_dataset)
    DECL_DL_TEMPLATELIST_READ(drawlist_id)
};

enum class violin_base_pos_t{
    left,
    middle,
    right
};
enum class violin_dir_t{
    left,
    right,
    left_right
};
enum class violin_scale_t{
    self,
    per_attribute,
    all,
    COUNT
};
const structures::enum_names<violin_scale_t> violin_scale_names{
    "self",
    "per attribute",
    "all attributes"
};
struct violin_appearance_t{
    violin_base_pos_t   base_pos{violin_base_pos_t::middle};
    violin_dir_t        dir{violin_dir_t::left_right};
    violin_scale_t      scale{violin_scale_t::per_attribute};
    ImVec4              color{1.f, 1.f, 1.f, 1.f};
    bool                span_full{false};
};

struct session_common{
    std::vector<attribute_order_info>                       attribute_order_infos{};
    mutable std::map<std::string_view, violin_appearance_t> attribute_violin_appearances{};
    std::map<std::string_view, bool>                        attribute_log{};
};
struct drawlist_session_state_t: public session_common{
    std::map<std::string_view, drawlist_info>   drawlists{};
    std::vector<std::string_view>               matrix_elements{};
};
struct attribute_session_state_t: public session_common{
    std::vector<drawlist_info>                  drawlists{};
};

struct settings_common{
    mutable ImVec4  plot_background{.0f, .0f, .0f, 1.f};
    mutable float   line_hover_dist{5.f};
    mutable float   line_thickness{1.f};
    mutable float   line_alpha{1.f};
    mutable float   area_alpha{.5f};
            int     histogram_bin_count{100};
            float   smoothing_std_dev{-1};
            bool    ignore_zero_bins{false};
    mutable float   plot_height{200};
    mutable float   plot_padding{5};
    mutable bool    reposition_attributes_on_update{false};
    mutable std::string attribute_color_palette_type{"Qualitative"};
    mutable std::string attribute_color_palette{"Set3"};
};
struct attribute_settings_t: public settings_common{

};
struct drawlist_settings_t: public settings_common{
    mutable std::array<int, 2> matrix_dimensions{2, 5};
};

//const std::array<std::string_view, 7> violin_positions{"left full", "left half", "middle left", "middle right", "middle both", "right full", "right half"};
const std::map<std::tuple<violin_base_pos_t, violin_dir_t, bool>, std::string_view> violin_positions{
    {{violin_base_pos_t::left, violin_dir_t::right, true}, "left full"},
    {{violin_base_pos_t::left, violin_dir_t::right, false}, "left half"},
    {{violin_base_pos_t::middle, violin_dir_t::left, false}, "middle left"},
    {{violin_base_pos_t::middle, violin_dir_t::right, false}, "middle right"},
    {{violin_base_pos_t::middle, violin_dir_t::left_right, false}, "middle both"},
    {{violin_base_pos_t::right, violin_dir_t::left, true}, "right full"},
    {{violin_base_pos_t::right, violin_dir_t::left, false}, "right half"}
};

struct local_storage{
    activation_tracker  active;
    bounds_tracker      bounds;
    color_tracker       color;
};
}
}

template<> struct std::hash<structures::violins::drawlist_attribute>{
    inline size_t operator()(const structures::violins::drawlist_attribute& da) const{
        size_t hash = std::hash<std::string_view>{}(da.dl);
        return std::hash_combine(hash, da.att);
    }
};