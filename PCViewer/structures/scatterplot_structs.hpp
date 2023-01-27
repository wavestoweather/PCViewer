#pragma once
#include <json_util.hpp>
#include <enum_names.hpp>
#include <drawlists.hpp>
#include <vk_util.hpp>
#include <imgui.h>
#include <default_hash.hpp>
#include <data_type.hpp>
#include <default_struct_equal.hpp>
#include <flat_set.hpp>
#include "../imgui_nodes/crude_json.h"

namespace structures{
namespace scatterplot_wb{
    enum class plot_type_t{
        matrix,
        list,
        COUNT
    };
    const structures::enum_names<plot_type_t> plot_type_names{
        "matrix",
        "list"
    };

    enum class data_source_t{
        array_t,
        histogram_t
    };

    struct settings_t{
        uint32_t    plot_width{400};   // width of 1 subplot
        uint32_t    sample_count{1};
        VkFormat    plot_format{VK_FORMAT_R16G16B16A16_UNORM};
        mutable double plot_padding{5};   // padding inbeteween the scatter plot images
        mutable ImVec4 plot_background_color{0, 0, 0, 1};
        plot_type_t plot_type{plot_type_t::list};
        std::array<int, 2> plot_matrix{1, 2};
        size_t      large_vis_threshold{500000};
        mutable float uniform_radius{5.f};

        bool operator==(const settings_t& o) const{
            COMP_EQ_OTHER(o, plot_width);
            COMP_EQ_OTHER(o, sample_count);
            COMP_EQ_OTHER(o, plot_padding);
            COMP_EQ_OTHER(o, plot_format);
            COMP_EQ_OTHER_VEC4(o, plot_background_color);
            COMP_EQ_OTHER(o, plot_type);
            COMP_EQ_OTHER(o, large_vis_threshold);
            COMP_EQ_OTHER(o, uniform_radius);
            return true;
        }
        settings_t() = default;
        settings_t(const crude_json::value& json){
            auto& t = *this;
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, plot_width, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, sample_count, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT(json, t, plot_padding);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, plot_format, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_VEC4(json, t, plot_background_color);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_ENUM_NAME(json, t, plot_type, plot_type_names);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, large_vis_threshold, double);
            JSON_ASSIGN_JSON_FIELD_TO_STRUCT_CAST(json, t, uniform_radius, double);
        }
        operator crude_json::value() const {
            auto& t = *this;
            crude_json::value json(crude_json::type_t::object);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, plot_width, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, sample_count, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON(json, t, plot_padding);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, plot_format, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_VEC4(json, t, plot_background_color);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_ENUM_NAME(json, t, plot_type, plot_type_names);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, large_vis_threshold, double);
            JSON_ASSIGN_STRUCT_FIELD_TO_JSON_CAST(json, t, uniform_radius, double);
            return json;
        }
    };

    enum class splat_form: uint32_t{
        circle,
        square,
        COUNT
    };
    const enum_names<splat_form> splat_form_names{
        "circle",
        "square"
    };

    struct scatterplot_dl_appearance{
        splat_form splat{};
        float      radius{5.f};
    };
    using tracked_dl_appearance = change_tracker<scatterplot_dl_appearance>;
    using appearance_tracker = change_tracker<drawlist::appearance>;
    struct drawlist_info{
        std::string_view                    drawlist_id;
        bool                                linked_with_drawlist;
        util::memory_view<appearance_tracker> appearance;
        tracked_dl_appearance               scatter_appearance;
        change_tracker<bool>                priority_render;

        bool any_change() const             {return appearance->changed;}
        void clear_changes()                {appearance->changed = false;}
        DECL_DRAWLIST_READ(drawlist_id)
        DECL_DRAWLIST_WRITE(drawlist_id)
        DECL_DATASET_READ(drawlist_read().parent_dataset)
        DECL_DATASET_WRITE(drawlist_read().parent_dataset)
        DECL_DL_TEMPLATELIST_READ(drawlist_id)
    };

    struct plot_data_t{
        image_info  image{};
        VkImageView image_view{};
        uint32_t    image_width{};
        VkFormat    image_format{};
        ImTextureID image_descriptor{}; // called desccriptor as it internally is, contains the descriptor set for imgui to render
        // samples are never stored here, as the sample count is only relevant for rendering, the image here always has only 1spp
    };

    struct plot_additional_data_t{
        std::string_view background_image{}; // only stores the string view to avoid problems when deleting the image
        ImVec2           left_top{}; // describes the pos in attribute values of the left top corner
        ImVec2           right_bot{};// describes the pos in attirbute values of the right bot corener
    };

    struct attribute_pair{
        std::string_view a;
        std::string_view b;

        bool operator==(const attribute_pair& o) const {return a == o.a && b == o.b;}
        bool operator!=(const attribute_pair& o) const {return a != o.a || b != o.b;}
        operator bool() const {return a.size() && b.size();}
    };

    struct att_pair_dls{
        attribute_pair              atts;
        flat_set<std::string_view>  dls;

        bool operator==(const att_pair_dls& o) const {return atts == o.atts && dls == o.dls;}
        bool operator!=(const att_pair_dls& o) const {return *this != o;}
        operator bool() const {return atts && dls.size();}
    };

    struct pipeline_data{
        VkPipeline          pipeline{};
        VkPipelineLayout    pipeline_layout{};
    };

    struct output_specs{
        VkFormat                format{};
        VkSampleCountFlagBits   sample_count{};
        uint32_t                width{};
        data_type_t             data_type{};
        data_source_t           data_source{};
        VkRenderPass            render_pass{};

        DEFAULT_EQUALS(output_specs);
    };

    struct framebuffer_key{
        VkFormat                format;
        VkSampleCountFlagBits   sample_counts;
        uint32_t                width;

        DEFAULT_EQUALS(framebuffer_key);
    };

    struct framebuffer_val{
        image_info  image{};
        VkImageView image_view{};
        image_info  multisample_image{};
        VkImageView multisample_image_view{};
        VkRenderPass render_pass{};
        VkFramebuffer framebuffer{};
    };

    struct local_attribute_storage{
        activation_tracker  active;
        bounds_tracker      bounds;
        color_tracker       color;
    };
}
}

DEFAULT_HASH(structures::scatterplot_wb::attribute_pair);
DEFAULT_HASH(structures::scatterplot_wb::output_specs);
DEFAULT_HASH(structures::scatterplot_wb::framebuffer_key);
template<> struct std::hash<structures::scatterplot_wb::att_pair_dls>{
    size_t operator()(const structures::scatterplot_wb::att_pair_dls& o) const{
        size_t seed = hash<structures::scatterplot_wb::attribute_pair>{}(o.atts);
        return hash_combine(seed, util::memory_view<const std::string_view>(o.dls.data(), o.dls.size()).data_hash());
    }
};