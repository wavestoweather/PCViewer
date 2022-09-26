#pragma once
#include <string_view>
#include <memory_view.hpp>
#include <atomic>
#include <memory>

namespace structures{
struct workbench{
    // attribute to indicate if the workbench should be shown
    bool                active{false};
    const std::string   id;

    workbench(std::string_view id): id(id) {};
    // method to show the imgui window
    virtual void show() = 0;
};

struct dataset_dependency{
    struct update_flags{
        bool fragmented_update: 1;
        bool fragmented_first: 1;
        bool fragmented_last: 1;
    };
    virtual void add_dataset(std::string_view dataset_id) = 0;
    virtual void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags) = 0;
    virtual void remove_dataset(std::string_view dataset_id) = 0;
};

struct drawlist_dataset_dependency: public dataset_dependency{
    virtual void add_drawlist(std::string_view drawlist_id) = 0;
    virtual void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids) = 0;
    virtual void remove_drawlist(std::string_view drawlist_id) = 0;
};
}

namespace globals{
using unique_workbench = std::unique_ptr<structures::workbench>;
using workbenches_t = std::vector<unique_workbench>;
extern workbenches_t workbenches;
extern structures::workbench* primary_workbench;
extern structures::workbench* secondary_workbench;
using dataset_dependencies_t = std::vector<structures::dataset_dependency*>;
extern dataset_dependencies_t dataset_dependencies;
using drawlist_dataset_dependencies_t = std::vector<structures::drawlist_dataset_dependency*>;
extern drawlist_dataset_dependencies_t drawlist_dataset_dependencies;
}