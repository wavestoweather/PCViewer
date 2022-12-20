#pragma once

#include <workbench_base.hpp>
#include <scatterplot_structs.hpp>
#include <change_tracker.hpp>
#include <robin_hood.h>

namespace workbenches{
class scatterplot_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using registered_histogram = structures::histogram_registry::scoped_registrator_t;
    using plot_type_t = structures::scatterplot_wb::plot_type_t;
    const structures::enum_names<plot_type_t>& plot_type_names = structures::scatterplot_wb::plot_type_names;

    std::vector<std::unique_ptr<appearance_tracker>>    _appearance_storage; // used for unlinked drawlists
    robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>> _registered_histograms;

    void _update_registered_histograms();
    void _update_plot_images();
public:
    using settings_t = structures::scatterplot_wb::settings_t;
    using drawlist_info = structures::scatterplot_wb::drawlist_info;
    using plot_data_t = structures::scatterplot_wb::plot_data_t;
    using attribute_order_info = structures::workbenches::attribute_order_info;
    using attribute_pair = structures::scatterplot_wb::attribute_pair;
    template<typename T> using changing_vector = structures::change_tracker<std::vector<T>>;

    structures::change_tracker<settings_t>                  settings{};
    structures::change_tracker<std::vector<drawlist_info>>  drawlist_infos{};
    robin_hood::unordered_map<attribute_pair, plot_data_t>  plot_datas{};
    changing_vector<structures::attribute>                  attributes{};
    changing_vector<attribute_order_info>                   attribute_order_infos{};
    changing_vector<attribute_pair>                         plot_list{};    // is used when plot_type is switched to list mode

    scatterplot_workbench(std::string_view id);

    // workbench override methods
    void notify_drawlist_dataset_update() override;
    void show() override;

    void                set_settings(const crude_json::value& json) override {settings = settings_t(json);}
    crude_json::value   get_settings() const override {return settings.read();}
    void                set_session_data(const crude_json::value& json) override {};
    crude_json::value   get_session_data() const override {return {};}

    // drawlist_dataset_dependency methods
    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override {}
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override {}

    void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override {}
    void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override {}

    std::vector<uint32_t> get_active_ordered_indices();
};
}