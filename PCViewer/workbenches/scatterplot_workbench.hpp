#pragma once

#include <workbench_base.hpp>
#include <scatterplot_structs.hpp>
#include <robin_hood.h>

// For the scatterplot workbench the scatterplot drawings are always assumed that the primary attribute (att.a)
// corresponds to the y axis, while the b attribute is rendered on  the x axis
// For the rendering this has to be taken into account...
namespace workbenches{
class scatterplot_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
public:
    using settings_t = structures::scatterplot_wb::settings_t;
    using drawlist_info = structures::scatterplot_wb::drawlist_info;
    using plot_data_t = structures::scatterplot_wb::plot_data_t;
    using plot_additional_data_t = structures::scatterplot_wb::plot_additional_data_t;
    using attribute_order_info = structures::attribute_info;
    using attribute_pair = structures::scatterplot_wb::attribute_pair;
    using att_pair_dls = structures::scatterplot_wb::att_pair_dls;
    template<typename T> using changing_vector = structures::change_tracker<std::vector<T>>;
    using const_attribute_info_ref = std::reference_wrapper<const attribute_order_info>;
private:
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using registered_histogram = structures::histogram_registry::scoped_registrator_t;
    using plot_type_t = structures::scatterplot_wb::plot_type_t;
    const structures::enum_names<plot_type_t>& plot_type_names = structures::scatterplot_wb::plot_type_names;
    using splat_form_t = structures::scatterplot_wb::splat_form;
    const structures::enum_names<splat_form_t>& splat_form_names = structures::scatterplot_wb::splat_form_names;
    using local_attribute_storage = std::unique_ptr<structures::scatterplot_wb::local_attribute_storage>;

    std::vector<std::unique_ptr<appearance_tracker>>    _appearance_storage; // used for unlinked drawlists
    robin_hood::unordered_map<std::string_view, local_attribute_storage> _local_attribute_storage; 
    robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>> _registered_histograms;

    bool                _drawlists_updated{};
    std::vector<float>  _plot_x_vals{};
    att_pair_dls        _started_lasso_attributes{};
    ImVec2              _last_lasso_point{};
    att_pair_dls        _popup_attributes{};
    bool                _request_registrators_update{};

    std::string         _att_pair_regex{};
    bool                _regex_error{};
    std::vector<att_pair_dls> _matrix_scatterplots{};

    void _update_registered_histograms();
    void _update_plot_images();
    void _update_plot_list();
    void _render_plot();
    void _update_attribute_order_infos();
    void _show_general_settings();
public:
    structures::change_tracker<settings_t>                  settings{};
    structures::change_tracker<std::vector<drawlist_info>>  drawlist_infos{};
    robin_hood::unordered_map<att_pair_dls, plot_data_t>    plot_datas{};
    robin_hood::unordered_map<att_pair_dls, plot_additional_data_t> plot_additional_datas{};
    std::vector<plot_additional_data_t>                     plot_additional_datas_matrix{};     // needed as for matrix elements without attributes and drawlists no background can exist
    changing_vector<attribute_order_info>                   attribute_order_infos{};
    changing_vector<att_pair_dls>                           plot_list{};    // only contains the needed attribute pairs

    scatterplot_workbench(std::string_view id);

    // workbench override methods
    void show() override;

    void                set_settings(const crude_json::value& json) override {settings = settings_t(json);}
    crude_json::value   get_settings() const override {return settings.read();}
    void                set_session_data(const crude_json::value& json) override {};
    crude_json::value   get_session_data() const override {return {};}

    // drawlist_dataset_dependency methods
    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override {}
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override {}
    void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const structures::gpu_sync_info& sync_info = {}) override;

    void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;

    std::vector<const_attribute_info_ref> get_active_ordered_attributes() const;
    bool                        all_registrators_updated(bool rendered = false) const;
    const attribute_order_info& get_attribute_order_info(std::string_view attribute) const;
};
}