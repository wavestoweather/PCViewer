#pragma once
#include <string_view>
#include <memory_view.hpp>
#include <atomic>


namespace structures{
struct workbench{
    // attribuete to indicate if the workbench should be shown
    bool active{false};

    // method to show the imgui window
    virtual void show() = 0;
};

struct dataset_dependency{
    struct update_flags{
        bool fragmented_update: 1;
        bool fragmented_first: 1;
        bool fragmented_last: 1;
    };
    virtual void addDataset(std::string_view datasetId) = 0;
    virtual void signalDatasetUpdate(const util::memory_view<std::string_view>& datasetIds, update_flags flags) = 0;
    virtual void removeDataset(std::string_view datasetId) = 0;
};

struct drawlist_dependency: public dataset_dependency{
    virtual void addDrawlist(std::string_view drawlistId) = 0;
    virtual void signalDrawlistUpdate(const util::memory_view<std::string_view>& drawlistIds) = 0;
    virtual void removeDrawlist(std::string_view drawlistId) = 0;
};
}