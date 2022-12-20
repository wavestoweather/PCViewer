#pragma once
#include <change_tracker.hpp>
#include <drawlists.hpp>
#include <scatterplot_structs.hpp>

namespace workbenches{
    class scatterplot_workbench;
}

namespace pipelines{
class scatterplot_renderer{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using output_specs      = structures::scatterplot_wb::output_specs;
    using pipeline_data     = structures::scatterplot_wb::pipeline_data;
    using time_point        = std::chrono::time_point<std::chrono::system_clock>;
    using framebuffer_key   = structures::scatterplot_wb::framebuffer_key;

    struct push_constants{
        VkDeviceAddress data_header_address;
        VkDeviceAddress index_buffer_address;
        VkDeviceAddress activation_bitset_address;
        uint32_t        attribute_a;
        uint32_t        attribute_b;
        float           a_min;
        float           a_max;
        float           b_min;
        float           b_max;
        uint32_t        flip_axes;
    };
    struct push_constants_large_vis{
        VkDeviceAddress counts_address;
        uint32_t        flip_axes;
    };

    const std::string_view vertex_shader_path{"shader/scatterplot_renderer.vert.spv"};
    const std::string_view large_vis_vertex_shader_path{"shader/scatterplot_renderer.vert.spv"};
    const std::string_view fragment_shader_path{"shader/scatterplot_renderer.vert.spv"};

    robin_hood::unordered_map<output_specs, pipeline_data>  _pipelines{};
    robin_hood::unordered_map<VkPipeline, time_point>       _pipeline_last_use{};
    robin_hood::unordered_map<framebuffer_key, VkFramebuffer>_framebuffers{};
    robin_hood::unordered_map<VkFramebuffer, time_point>    _framebuffer_last_use{};

    // vulkan resources that are the same for all drawlists/parallel_coordinates_windows
    structures::buffer_info                                 _attribute_info_buffer{};
    VkCommandPool                                           _command_pool{};
    VkFence                                                 _render_fence{};    // needed as only a single attribute info buffer exists
    std::vector<VkCommandBuffer>                            _render_commands{};

    struct multisample_image_t{
        structures::image_info  image;
        VkImageView             image_view;
        VkFramebuffer           framebuffer;
        VkFormat                format;
        VkSampleCountFlagBits   spp;
        uint32_t                width;
    }   _multisample_image{};   // currently supports only a single multisample image

    const pipeline_data& _get_or_create_pipeline(const output_specs& output_specs);
    VkFramebuffer        _get_or_create_framebuffer(VkRenderPass render_pass, uint32_t width, VkImageView dst_image, VkImageView multisample_image = {});

    scatterplot_renderer();
public:
    using drawlist_info = structures::scatterplot_wb::drawlist_info;

    struct render_info{
        const workbenches::scatterplot_workbench& workbench;
    };

    scatterplot_renderer(const scatterplot_renderer&) = delete;
    scatterplot_renderer& operator=(const scatterplot_renderer&) = delete;

    static scatterplot_renderer& instance();
    void render(const render_info& info);

    uint32_t max_pipeline_count{20};
    uint32_t max_framebuffer_count{100};
};
}