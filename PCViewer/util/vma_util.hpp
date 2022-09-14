#pragma once
#include <vk_mem_alloc.h>
#include <vk_context.hpp>
#include <memory_view.hpp>

namespace util{
namespace vma{
inline void upload_data(memory_view<const uint8_t> data, structures::buffer_info dst){
    void* mapped;
    auto res = vmaMapMemory(globals::vk_context.allocator, dst.allocation, &mapped); util::check_vk_result(res);
    std::memcpy(mapped, data.data(), data.byteSize());
    vmaUnmapMemory(globals::vk_context.allocator, dst.allocation);
}
}
}