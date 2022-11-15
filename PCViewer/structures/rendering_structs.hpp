#pragma once
#include <enum_names.hpp>
#include <memory_view.hpp>
#include <vulkan/vulkan.h>
#include <optional>
#include <drawlists.hpp>
#include <data_type.hpp>

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
        COUNT
    };
    const structures::enum_names<render_type> render_type_names{
        "polyline_spline",
        "large_vis_lines",
        "large_vis_density"
    };

    struct output_specs{
        VkImageView             plot_image_view{};
        VkFormat                format{};
        VkSampleCountFlagBits   sample_count{};
        uint32_t                width{};
        uint32_t                height{};
        render_type             render_typ{};
        data_type               data_typ{};

        bool operator==(const output_specs& o) const {return util::memory_view<const uint32_t>(util::memory_view(*this)).equal_data(util::memory_view<const uint32_t>(util::memory_view(o)));};
    };

    struct pipeline_data{
        VkPipeline              pipeline{};
        VkPipelineLayout        pipeline_layout{};
        VkRenderPass            render_pass{};
        VkFramebuffer           framebuffer{};  
        structures::image_info  multi_sample_image{};
        VkImageView             multi_sample_view{};
    };

    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using median_tracker = structures::change_tracker<structures::median_type>;
    struct drawlist_info{
        std::string_view                        drawlist_id;
        bool                                    linked_with_drawlist;
        util::memory_view<appearance_tracker>   appearance;
        util::memory_view<median_tracker>       median;

        bool any_change() const                             {return appearance->changed || median->changed;}
        void clear_change()                                 {appearance->changed = false; median->changed = false;}
        const structures::drawlist& drawlist_read() const   {return globals::drawlists.read().at(drawlist_id).read();}
        structures::drawlist& drawlist_write() const        {return globals::drawlists()[drawlist_id]();}
    };
}

namespace brusher{
    struct pipeline_specs{
        data_type data_typ;

        bool operator==(const pipeline_specs& o) const {return data_typ == o.data_typ;}
    };
    struct pipeline_data{
        VkPipeline          pipeline{};
        VkPipelineLayout    pipeline_layout{};
    };
}
}

template<> struct std::hash<structures::parallel_coordinates_renderer::output_specs>{
    size_t operator()(const structures::parallel_coordinates_renderer::output_specs & x) const
    {
        util::memory_view<const uint32_t> as_uint = util::memory_view<const uint32_t>(util::memory_view<const structures::parallel_coordinates_renderer::output_specs>(x));
        size_t seed = 0;
        for(uint32_t i: as_uint)
            seed = std::hash_combine(seed, i);
        return seed;
    }
};

template<> struct std::hash<structures::brusher::pipeline_specs>{
    size_t operator()(const structures::brusher::pipeline_specs & x) const
    {
        util::memory_view<const uint32_t> as_uint = util::memory_view<const uint32_t>(util::memory_view<const structures::brusher::pipeline_specs>(x));
        size_t seed = 0;
        for(uint32_t i: as_uint)
            seed = std::hash_combine(seed, i);
        return seed;
    }
};
