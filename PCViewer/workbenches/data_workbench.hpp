#pragma once
#include <workbench_base.hpp>
#include <dataset_convert_data.hpp>

namespace workbenches{

// main data workbench which exists exactly once and is not closable
class data_workbench: public structures::workbench, public structures::drawlist_dataset_dependency{
    std::string_view                _popup_ds_id{};
    std::string_view                _popup_tl_id{};
    
    structures::dataset_convert_data _ds_convert_data{};

public:
    data_workbench(std::string_view id): workbench(id) {};

    void show() override;

    void addDataset(std::string_view datasetId) override {};
    void signalDatasetUpdate(const util::memory_view<std::string_view>& datasetIds, update_flags flags) override {};
    void removeDataset(std::string_view datasetId) override {};

    void addDrawlist(std::string_view drawlistId) override {};
    void signalDrawlistUpdate(const util::memory_view<std::string_view>& drawlistIds) override {};
    void removeDrawlist(std::string_view drawlistId) override {};
};

}