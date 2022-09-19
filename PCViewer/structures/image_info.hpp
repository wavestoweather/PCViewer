#pragma once
#include <vk_mem_alloc.h>

namespace structures{
struct image_info{
    VkImage         image{};
    VmaAllocation   allocation{};

    bool operator==(const image_info& o)    const {return image == o.image && allocation == o.allocation;}
    operator bool()                         const {return image && allocation;}
};
}
