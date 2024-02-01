#pragma once
#include <string_view>
#include <memory_view.hpp>
#include <atomic>
#include <memory>
#include <gpu_sync.hpp>
#include "../imgui_nodes/crude_json.h"

typedef struct VkSemaphore_T *VkSemaphore;
typedef uint32_t VkFlags;
typedef VkFlags VkSemaphoreWaitFlags;

namespace structures{

struct workbench{
    // attribute to indicate if the workbench should be shown
    bool                active{false};
    const std::string   id;

    workbench(std::string_view id): id(id) {};
    virtual ~workbench() {};
    // method to show the imgui window
    virtual void show() = 0;
    
    // methods required to load/save state of the workbench
    virtual void                set_settings(const crude_json::value& json) {};
    virtual crude_json::value   get_settings() const {return {};};
    virtual void                set_session_data(const crude_json::value& json) {};
    virtual crude_json::value   get_session_data() const {return {};};
};

struct dataset_dependency{
    struct update_flags{
        bool fragmented_update: 1;
        bool fragmented_first: 1;
        bool fragmented_last: 1;
    };
    virtual void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const gpu_sync_info& sync_info = {}) {};
    virtual ~dataset_dependency() {}
};

struct drawlist_dataset_dependency: public dataset_dependency{
    virtual void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const gpu_sync_info& sync_info = {}) {};
    virtual ~drawlist_dataset_dependency() {}
};
}

namespace globals{
using unique_workbench = std::unique_ptr<structures::workbench>;
using workbenches_t = std::vector<unique_workbench>;
extern workbenches_t workbenches;
extern structures::workbench* primary_workbench;
extern structures::workbench* secondary_workbench;
using dataset_dependencies_t = std::vector<structures::dataset_dependency*>;
extern dataset_dependencies_t dataset_dependencies;                     // also contains all drawlist_dataset_dependencies as they are a specialization
using drawlist_dataset_dependencies_t = std::vector<structures::drawlist_dataset_dependency*>;
extern drawlist_dataset_dependencies_t drawlist_dataset_dependencies;
using workbench_index_t = std::map<std::string_view, structures::workbench&>;
extern workbench_index_t workbench_index;
}