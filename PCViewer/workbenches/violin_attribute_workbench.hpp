#pragma once
#include <workbench_base.hpp>
#include <violin_structures.hpp>
#include <robin_hood.h>

namespace workbenches{
class violin_attribute_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
public:
    using drawlist_attribute = structures::violins::drawlist_attribute;
private:
    using appearance_tracker = structures::change_tracker<structures::drawlist::appearance>;
    using registered_histogram = structures::histogram_registry::scoped_registrator_t;
    using registered_histogram_map = robin_hood::unordered_map<std::string_view, std::vector<registered_histogram>>;
    using appearance_storage_t = std::vector<std::unique_ptr<appearance_tracker>>;
    using local_storage_t = std::vector<std::unique_ptr<structures::violins::local_storage>>;
    using drawlist_attribute_histograms_t = robin_hood::unordered_map<drawlist_attribute, structures::violins::histogram>; 

    appearance_storage_t        _appearance_storage; // used for unlinked drawlists
    local_storage_t             _local_storage;
    registered_histogram_map    _registered_histograms;
    drawlist_attribute_histograms_t _drawlist_attribute_histograms;
    std::map<std::string_view, float> _per_attribute_max{};
    float                       _global_max{};
    drawlist_attribute          _hovered_dl_attribute{};

    void _update_attribute_histograms();
    void _update_registered_histograms();
    void _update_attribute_order_infos();
public:
    using attribute_settings_t = structures::change_tracker<structures::violins::attribute_settings_t>;
    using attribute_session_state_t = structures::change_tracker<structures::violins::attribute_session_state_t>;
    using const_attribute_info_ref = std::reference_wrapper<const structures::attribute_info>;
    attribute_settings_t        settings{};
    attribute_session_state_t   session_state{};

    violin_attribute_workbench(std::string_view id);

    void show() override;

    void                set_settings(const crude_json::value& json) {};
    crude_json::value   get_settings() const {return {};};
    void                set_session_data(const crude_json::value& json) override {}
    crude_json::value   get_session_data() const {return {};}

    // drawlist_dataset_dependency methods
    void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override {}
    void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const structures::gpu_sync_info& sync_info = {}) override {}
    void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const structures::gpu_sync_info& sync_info = {}) override;

    void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;
    void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const structures::gpu_sync_info& sync_info = {}) override;

    std::vector<drawlist_attribute> get_active_ordered_drawlist_attributes() const;
    std::vector<const_attribute_info_ref> get_active_ordered_attributes() const;
    std::map<std::string_view, structures::min_max<float>>  get_active_attribute_bounds() const;
};
}