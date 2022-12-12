#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <parallel_coordinates_renderer.hpp>
#include <enum_names.hpp>

namespace workbenches{

class parallel_coordinates_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
public:
    struct attribute_order_info{
        uint32_t    attribut_index{};
        bool        active{true};

        bool operator==(const attribute_order_info& o) const {return attribut_index == o.attribut_index && active == o.active;}
    };
private:
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using drawlist_info = structures::parallel_coordinates_renderer::drawlist_info;
    using registered_histogram = structures::histogram_registry::scoped_registrator_t;

    // both are unique_ptrs to avoid issues with the memory_views when data elements are deleted in the vector
    std::vector<std::unique_ptr<appearance_tracker>>         _storage_appearance;
    std::vector<std::unique_ptr<structures::median_type>>    _storage_median_type;
    robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>> _registered_histograms;
    robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>> _registered_axis_histograms;
    bool                                                     _select_priority_center_single{false};
    bool                                                     _select_priority_center_all{false};

    void _update_plot_image();
    void _draw_setting_list();
    void _swap_attributes(const attribute_order_info& from, const attribute_order_info& to);
    void _update_registered_histograms(bool request_update = false);
public:
    enum histogram_type: uint32_t{
        none,
        line_histogram,
        grey_scale_histogram,
        heatmap_histogram,
        COUNT
    };
    const structures::enum_names<histogram_type> histogram_type_names{
        "none",
        "line histogram",
        "grey scale histogram",
        "heatmap histogram"
    };
    struct settings{
        // mutable to be accessed and change even when const (can not be tracked)
        mutable bool        enable_axis_lines{true};
        mutable bool        min_max_labes{false};
        mutable bool        axis_tick_label{};
        mutable std::string axis_tick_fmt{"%6.4g"};
        mutable int         axis_tick_count{0};
        mutable size_t      render_batch_size{};
        mutable float       brush_box_width{20};
        mutable float       brush_box_border_width{2};
        mutable float       brush_box_border_hover_width{5};
        mutable ImVec4      brush_box_global_color{.2f, 0, .8f, 1};
        mutable ImVec4      brush_box_local_color{1, 0, .1f, 1};
        mutable ImVec4      brush_box_selected_color{.8f, .8f, 0, 1};
        mutable float       brush_arrow_button_move{.01f};
        mutable float       brush_drag_threshold{.5f};
        mutable int         live_brush_threshold{5000000};

        // when these are changed the whole data plot has to be rendered
        histogram_type hist_type{};
        float       histogram_blur_width{.01f};
        float       histogram_width{.01f};
        ImVec4      plot_background{0,0,0,1};
        int         histogram_rendering_threshold{500000};
        bool        render_splines{};
    };
    struct plot_data{
        uint32_t                width{2000};
        uint32_t                height{480};
        structures::image_info  image{};
        VkImageView             image_view{};
        VkSampleCountFlagBits   image_samples{VK_SAMPLE_COUNT_8_BIT};
        VkFormat                image_format{VK_FORMAT_R16G16B16A16_UNORM};
        ImTextureID             image_descriptor{}; // called descriptor as internally it is a descriptor
    };
    enum class render_strategy{
        all,
        batched,
        COUNT
    };
    const structures::enum_names<render_strategy> render_strategy_names{
        "all",
        "batched",
    };
    const std::array<VkFormat, 4>                           available_formats{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};

    structures::change_tracker<settings>                    setting{};
    structures::change_tracker<std::vector<drawlist_info>>  drawlist_infos{};     // the order here is the render order of the drawlists
    structures::alpha_mapping_type                          alpha_mapping_typ{};
    structures::change_tracker<plot_data>                   plot_data{};
    structures::change_tracker<std::vector<structures::attribute>> attributes{};
    structures::change_tracker<std::vector<attribute_order_info>> attributes_order_info{};
    render_strategy                                         render_strategy{};

    parallel_coordinates_workbench(const std::string_view id);

    void render_plot();
    std::vector<uint32_t> get_active_ordered_indices() const;

    // overriden methods
    void show() override;

    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override{};
    void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, structures::dataset_dependency::update_flags flags, const structures::gpu_sync_info& sync_info = {}) override{};
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
};

}