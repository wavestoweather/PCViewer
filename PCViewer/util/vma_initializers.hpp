#pragma once
#include <vk_mem_alloc.h>

namespace util{
namespace vma{
namespace initializers{
inline VmaAllocationCreateInfo allocationCreateInfo(VmaAllocationCreateFlags flags = {}, VmaMemoryUsage usage = VMA_MEMORY_USAGE_AUTO){
    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags = flags;
    allocationCreateInfo.usage = usage;
    return allocationCreateInfo;
}
}
}
}