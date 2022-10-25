#pragma once
#include <memory_view.hpp>
#include <vulkan/vulkan.h>

namespace structures{
struct gpu_timing_info{
    VkQueryPool query_pool{};
    uint32_t    query_pool_size{};
    uint32_t    cur_query_index{};
};
}