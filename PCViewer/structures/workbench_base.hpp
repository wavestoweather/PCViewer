#pragma once
#include <string_view>
#include <memory_view.hpp>
#include <atomic>
#include <memory>

typedef struct VkSemaphore_T *VkSemaphore;
typedef uint32_t VkFlags;
typedef VkFlags VkSemaphoreWaitFlags;

namespace structures{
struct gpu_sync_info{
    util::memory_view<VkSemaphore>          wait_semaphores{};
    util::memory_view<VkSemaphoreWaitFlags> wait_flags{};
    util::memory_view<VkSemaphore>          signale_semaphores{};
};

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
    virtual void add_datasets(const util::memory_view<std::string_view>& dataset_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void signal_dataset_update(const util::memory_view<std::string_view>& dataset_ids, update_flags flags, const gpu_sync_info& sync_info = {}) = 0;
    virtual void remove_datasets(const util::memory_view<std::string_view>& dataset_ids, const gpu_sync_info& sync_info = {}) = 0;
};

struct drawlist_dataset_dependency: public dataset_dependency{
    virtual void add_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void signal_drawlist_update(const util::memory_view<std::string_view>& drawlist_ids, const gpu_sync_info& sync_info = {}) = 0;
    virtual void remove_drawlists(const util::memory_view<std::string_view>& drawlist_ids, const gpu_sync_info& sync_info = {}) = 0;
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