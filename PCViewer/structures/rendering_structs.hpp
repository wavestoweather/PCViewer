#pragma once
#include <enum_names.hpp>
#include <memory_view.hpp>
#include <vulkan/vulkan.h>
#include <optional>
#include <drawlists.hpp>
#include <data_type.hpp>
#include <default_hash.hpp>

namespace structures{

enum class alpha_mapping_type: uint32_t{
    multiplicative,
    bound01,
    const_alpha,
    alpha_adoption,
    COUNT
};
static enum_names<alpha_mapping_type> alpha_mapping_type_names{
    "multiplicative",
    "bound01",
    "const_alpha",
    "alpha_adoption"
};

namespace parallel_coordinates_renderer{
    enum class render_type: uint32_t{
        polyline_spline,
        large_vis_lines,
        large_vis_density,
        histogram,
        COUNT
    };
    const structures::enum_names<render_type> render_type_names{
        "polyline_spline",
        "large_vis_lines",
        "large_vis_density",
        "histogram"
    };

    struct output_specs{
        VkImageView             plot_image_view{};
        VkFormat                format{};
        VkSampleCountFlagBits   sample_count{};
        uint32_t                width{};
        uint32_t                height{};
        render_type             render_typ{};
        data_type_t             data_typ{};

        bool operator==(const output_specs& o) const {return util::memory_view<const uint32_t>(util::memory_view(*this)).equal_data(util::memory_view<const uint32_t>(util::memory_view(o)));}
    };

    struct pipeline_data{
        VkPipeline              pipeline{};
        VkPipelineLayout        pipeline_layout{};
        VkRenderPass            render_pass{};
        VkFramebuffer           framebuffer{};  
    };

    struct multisample_key{
        VkFormat                format{};
        VkSampleCountFlagBits   sample_count{};
        uint32_t                width{};
        uint32_t                height{};

        bool operator==(const multisample_key& o) const {return util::memory_view<const uint32_t>(util::memory_view(*this)).equal_data(util::memory_view<const uint32_t>(util::memory_view(o)));}
    };
    struct multisample_val{
        uint32_t                count{};
        structures::image_info  image{};
        VkImageView             image_view{};
    };

    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using median_tracker = structures::change_tracker<structures::median_type>;
    struct drawlist_info{
        std::string_view                        drawlist_id;
        bool                                    linked_with_drawlist;
        util::memory_view<appearance_tracker>   appearance;
        util::memory_view<median_tracker>       median;
        change_tracker<bool>                    priority_render;

        bool any_change() const {return appearance->changed || median->changed || priority_render.changed;}
        void clear_change()     {appearance->changed = false; median->changed = false; priority_render.changed = false;}
        DECL_DRAWLIST_READ(drawlist_id)
        DECL_DRAWLIST_WRITE(drawlist_id)
        DECL_DATASET_READ(drawlist_read().parent_dataset)
        DECL_DATASET_WRITE(drawlist_read().parent_dataset)
        DECL_DATASET_NO_TRACK(drawlist_read().parent_dataset)
    };
}

namespace brusher{
    struct pipeline_specs{
        data_type_t data_type;

        bool operator==(const pipeline_specs& o) const {return data_type == o.data_type;}
    };
    struct pipeline_data{
        VkPipeline          pipeline{};
        VkPipelineLayout    pipeline_layout{};
    };
}
}

DEFAULT_HASH(structures::parallel_coordinates_renderer::output_specs);
DEFAULT_HASH(structures::brusher::pipeline_specs);
DEFAULT_HASH(structures::parallel_coordinates_renderer::multisample_key);
