#pragma once
#include <workbench_base.hpp>
#include <memory_view.hpp>
#include <imgui.h>
#include <parallel_coordinates_renderer.hpp>
#include <enum_names.hpp>

namespace workbenches{

class parallel_coordinates_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using drawlist_info = structures::parallel_coordinates_renderer::drawlist_info;
    using registered_histogram = structures::histogram_registry::scoped_registrator_t;
    using attribute_order_info_t = structures::attribute_info;
    using const_attribute_info_ref = std::reference_wrapper<const attribute_order_info_t>;

    // both are unique_ptrs to avoid issues with the memory_views when data elements are deleted in the vector
    std::vector<std::unique_ptr<appearance_tracker>>        _storage_appearance;
    std::vector<std::unique_ptr<structures::median_type>>   _storage_median_type;
    std::vector<std::unique_ptr<bool>>                      _storage_activation;
    robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>> _registered_histograms;
    robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>> _registered_axis_histograms;
    bool                                                     _select_priority_center_single{false};
    bool                                                     _select_priority_center_all{false};
    bool                                                     _dl_added{};

    void _update_plot_image();
    void _draw_setting_list();
    void _swap_attributes(const attribute_order_info_t& from, const attribute_order_info_t& to);
    void _update_registered_histograms(bool request_update = false);
    void _update_attribute_order_infos();
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
    struct settings_t{
        // mutable to be accessed and change even when const (can not be tracked)
        mutable bool        enable_axis_lines{true};
        mutable bool        min_max_labes{false};
        mutable bool        axis_tick_label{true};
        mutable bool        enable_category_labels{true};
        mutable std::string axis_tick_fmt{"%6.4g"};
        mutable int         axis_tick_count{5};
        mutable size_t      render_batch_size{};
        mutable double      brush_box_width{20};
        mutable double      brush_box_border_width{2};
        mutable double      brush_box_border_hover_width{5};
        mutable ImVec4      brush_box_global_color{.2f, 0, .8f, 1};
        mutable ImVec4      brush_box_local_color{1, 0, .1f, 1};
        mutable ImVec4      brush_box_selected_color{.8f, .8f, 0, 1};
        mutable double      brush_arrow_button_move{.01f};
        mutable double      brush_drag_threshold{.5f};
        mutable int         live_brush_threshold{5000000};

        // when these are changed the whole data plot has to be rendered
        histogram_type hist_type{};
        double      histogram_blur_width{.01f};
        double      histogram_width{.01f};
        ImVec4      plot_background{0,0,0,1};
        int         histogram_rendering_threshold{500000};
        bool        render_splines{};

        settings_t() = default;
        settings_t(const crude_json::value& json);
        operator crude_json::value() const;
        bool operator==(const settings_t& o) const;
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

    structures::change_tracker<settings_t>                  setting{};
    structures::change_tracker<std::vector<drawlist_info>>  drawlist_infos{};     // the order here is the render order of the drawlists
    structures::alpha_mapping_type                          alpha_mapping_typ{};
    structures::change_tracker<plot_data>                   plot_data{};
    structures::change_tracker<std::vector<attribute_order_info_t>> attribute_order_infos{};
    render_strategy                                         render_strategy{};

    parallel_coordinates_workbench(const std::string_view id);

    void                            render_plot();
    std::vector<const_attribute_info_ref> get_active_ordered_attributes() const;
    const attribute_order_info_t&   get_attribute_order_info(std::string_view attribute) const;
    bool                            all_registrators_updated(bool rendered = false) const;

    // overriden methods
    void show() override;

    void                set_settings(const crude_json::value& settings) override {setting() = settings;}
    crude_json::value   get_settings() const override { return setting.read();}

    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override{};
    void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, structures::dataset_dependency::update_flags flags, const structures::gpu_sync_info& sync_info = {}) override;
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
};

}