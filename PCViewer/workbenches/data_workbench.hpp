#pragma once
#include <workbench_base.hpp>
#include <dataset_convert_data.hpp>

namespace workbenches{

// main data workbench which exists exactly once and is not closable
class data_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
    std::string_view                _popup_ds_id{};
    std::string_view                _popup_tl_id{};
    std::string                     _open_filename{};
    std::string                     _table_filter{};
    bool                            _regex_error{};
    float                           _uniform_alpha{.01f};
    
    structures::templatelist_convert_data _tl_convert_data{};
    structures::templatelist_split_data   _tl_split_data{};

public:
    data_workbench(std::string_view id): workbench(id) {};

    void show() override;

    void add_datasets(const util::memory_view<std::string_view>& datasetId, const structures::gpu_sync_info& sync_info = {}) override {};
    void signal_dataset_update(const util::memory_view<std::string_view>& datasetIds, update_flags flags, const structures::gpu_sync_info& sync_info = {}) override {};
    void remove_datasets(const util::memory_view<std::string_view>& datasetId, const structures::gpu_sync_info& sync_info = {}) override {};

    void add_drawlists(const util::memory_view<std::string_view>& drawlistId, const structures::gpu_sync_info& sync_info = {}) override {};
    void signal_drawlist_update(const util::memory_view<std::string_view>& drawlistIds, const structures::gpu_sync_info& sync_info = {}) override {};
    void remove_drawlists(const util::memory_view<std::string_view>& drawlistId, const structures::gpu_sync_info& sync_info = {}) override {};
};

}