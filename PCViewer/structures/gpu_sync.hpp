#pragma once
#include <memory_view.hpp>
#include <vulkan/vulkan.h>

namespace structures{
struct gpu_sync_info{
    util::memory_view<VkSemaphore>          wait_semaphores{};
    util::memory_view<VkPipelineStageFlags> wait_masks{};
    util::memory_view<VkSemaphore>          signal_semaphores{};
};
}