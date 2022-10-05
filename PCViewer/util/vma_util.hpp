#pragma once
#include <vk_mem_alloc.h>
#include <vk_context.hpp>
#include <memory_view.hpp>
#include <buffer_info.hpp>

namespace util{
namespace vma{
inline void upload_data(memory_view<const uint8_t> data, structures::buffer_info dst){
    void* mapped;
    auto res = vmaMapMemory(globals::vk_context.allocator, dst.allocation, &mapped); util::check_vk_result(res);
    std::memcpy(mapped, data.data(), data.byteSize());
    vmaUnmapMemory(globals::vk_context.allocator, dst.allocation);
}

inline size_t get_buffer_size(const structures::buffer_info& buffer){
    if(!buffer)
        return 0;
    VmaAllocationInfo alloc_info{};
    vmaGetAllocationInfo(globals::vk_context.allocator, buffer.allocation, &alloc_info);
    return alloc_info.size;
}
}
}