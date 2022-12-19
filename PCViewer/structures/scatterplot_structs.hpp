#pragma once
#include <json_util.hpp>
#include <enum_names.hpp>
#include <drawlists.hpp>
#include <vk_util.hpp>
#include <imgui.h>
#include "../imgui_nodes/crude_json.h"

namespace structures{
namespace scatterplot_wb{
    enum class plot_type_t{
        matrix,
        list,
        COUNT
    };
    structures::enum_names<plot_type_t> plot_type_names{
        "matrix",
        "list"
    };

    struct settings_t{
        uint32_t    plot_width{150};   // width of 1 subplot
        uint32_t    sample_count{1};
        plot_type_t plot_type{plot_type_t::matrix};
        double      plot_padding{5};   // padding inbeteween the scatter plot images
        ImVec4      plot_background_color{0, 0, 0, 1};

        bool operator==(const settings_t& o){
            COMP_EQ_OTHER(o, plot_width);
            COMP_EQ_OTHER(o, sample_count);
            COMP_EQ_OTHER(o, plot_type);
            COMP_EQ_OTHER(o, plot_padding);
            COMP_EQ_OTHER_VEC4(o, plot_background_color);
            return true;
        }
        settings_t() = default;
        settings_t(const crude_json::value& json){
            auto& t = *this;
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, plot_width, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, sample_count, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, plot_type, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, plot_padding);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, t, plot_background_color);
        }
        operator crude_json::value() const {
            auto& t = *this;
            crude_json::value json(crude_json::type_t::object);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, plot_width, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, sample_count, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, plot_type, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, plot_padding);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, t, plot_background_color);
            return json;
        }
    };

    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    struct drawlist_info{
        std::string_view                    drawlist_id;
        bool                                linked_with_drawlist;
        util::memory_view<appearance_tracker> appearance;

        bool any_change() const             {return appearance->changed;}
        void clear_changes()                {appearance->changed = false;}
        DECL_DRAWLIST_READ(drawlist_id)
        DECL_DRAWLIST_WRITE(drawlist_id)
        DECL_DATASET_READ(drawlist_read().parent_dataset)
        DECL_DATASET_WRITE(drawlist_read().parent_dataset)
    };

    struct plot_data_t{
        image_info  plot_image{};
        VkImageView image_view{};
        VkFormat    image_format{};
        ImTextureID image_descriptor{}; // called desccriptor as it internally is, contains the descriptor set for imgui to render
    };
}
}