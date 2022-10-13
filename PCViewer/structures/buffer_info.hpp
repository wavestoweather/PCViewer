#pragma once
#include <vk_mem_alloc.h>

namespace structures{
struct buffer_info{
    VkBuffer        buffer{};
    size_t          size{};
    VmaAllocation   allocation{};

    bool operator==(const buffer_info& o)   const {return buffer == o.buffer && allocation == o.allocation;}
    operator bool()                         const {return buffer && allocation;}
};
}