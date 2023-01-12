#pragma once
#include <memory_view.hpp>
#include <vk_context.hpp>
#include <robin_hood.h>
#include <change_tracker.hpp>
#include <rendering_structs.hpp>
#include <drawlists.hpp>
#include <optional>
#include <chrono>
#include <imgui.h>

namespace workbenches{
    class parallel_coordinates_workbench;
}

namespace pipelines{

class parallel_coordinates_renderer{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using output_specs      = structures::parallel_coordinates_renderer::output_specs;
    using pipeline_data     = structures::parallel_coordinates_renderer::pipeline_data;
    using multisample_key   = structures::parallel_coordinates_renderer::multisample_key;
    using multisample_val   = structures::parallel_coordinates_renderer::multisample_val;
    using time_point        = std::chrono::time_point<std::chrono::system_clock>;

    struct push_constants{
        VkDeviceAddress     attribute_info_address;
        VkDeviceAddress     data_header_address;
        VkDeviceAddress     priorities_address;
        VkDeviceAddress     index_buffer_address;
        VkDeviceAddress     index_order_address;
        VkDeviceAddress     activation_bitset_address;
        uint32_t            vertex_count_per_line;        // is at least as high as attribute_count (when equal, polyline rendering)
        float               padding;
        uint32_t            sp,ace;
        ImVec4              color;
    };
    struct push_constants_large_vis{
        uint64_t    attribute_info_address{};
        uint64_t    histogram_address{};
        uint64_t    ordering_address{};           // needed for order dependant rendering (eg. priority rendering)
        int         a_axis{};                     // holds the final axis position index (after activation, reordering) for the primary histogram axis
        int         b_axis{};                     // holds the final axis position index (after activation, reordering) for the secondary histogram axis
        int         c_axis{};
        int         d_axis{};

        uint32_t    a_size{};
        uint32_t    b_size{};
        uint32_t    c_size{};
        uint32_t    d_size{};
        float       padding{};
        uint32_t    priority_rendering{};         // 1 indicates priority rendering should be used
        ImVec4      color{};
        uint32_t    line_verts{2};
    };

    struct push_constants_hist_vert{
        ImVec4 bounds{};
    };
    struct push_constants_hist_frag{
        uint64_t histogram_address{};
        uint32_t bin_count{};
        uint32_t mapping_type{};
        ImVec4   color{};
        float    blur_radius{};
    };

    const std::string_view vertex_shader_path{"shader/parallel_coordinates_renderer.vert.spv"};
    const std::string_view large_vis_vertex_shader_path{"shader/parallel_coordinates_renderer_large.vert.spv"};
    const std::string_view fragment_shader_path{"shader/parallel_coordinates_renderer.frag.spv"};
    const std::string_view histogram_vertex_shader_path{"shader/quad_forward.vert.spv"};
    const std::string_view histogram_fragment_shader_path{"shader/parallel_coordinates_histogram.frag.spv"};

    const uint32_t         _spline_resolution = 20;

    // vulkan resources that are the same for all drawlists/parallel_coordinates_windows
    structures::buffer_info                                 _attribute_info_buffer{};
    VkCommandPool                                           _command_pool{};
    VkFence                                                 _render_fence{};    // needed as only a single attribute info buffer exists
    std::vector<VkCommandBuffer>                            _render_commands{};

    robin_hood::unordered_map<output_specs, pipeline_data>  _pipelines{};
    robin_hood::unordered_map<VkPipeline, time_point>       _pipeline_last_use{};
    robin_hood::unordered_map<multisample_key, multisample_val> _multisample_images{};

    const pipeline_data& get_or_create_pipeline(const output_specs& output_specs);
    const structures::buffer_info& get_or_resize_info_buffer(size_t byte_size);

    parallel_coordinates_renderer();

    void _pre_render_commands(VkCommandBuffer commands, const output_specs& output_specs, bool clear_framebuffer = false, const ImVec4& clear_color = {});
    void _post_render_commands(VkCommandBuffer commands, VkFence fence = {}, util::memory_view<VkSemaphore> wait_semaphores = {}, util::memory_view<VkSemaphore> signal_semaphores = {});

public:
    using drawlist_info = structures::parallel_coordinates_renderer::drawlist_info;

    struct render_info{
        const workbenches::parallel_coordinates_workbench&  workbench;
        util::memory_view<VkSemaphore>                      wait_semaphores;
        util::memory_view<VkSemaphore>                      signal_semaphores;  
    };

    parallel_coordinates_renderer(const parallel_coordinates_renderer&) = delete;
    parallel_coordinates_renderer& operator=(const parallel_coordinates_renderer&) = delete;

    static parallel_coordinates_renderer& instance();

    void render(const render_info& info);

    uint32_t max_pipeline_count{20};
};
    
};
