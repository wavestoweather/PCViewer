#pragma once

#include <data.hpp>
#include <vk_context.hpp>
#include <memory_view.hpp>
#include <vk_context.hpp>
#include <vk_util.hpp>
#include <ranges.hpp>
#include <array_struct.hpp>
#include <vector>

namespace util{
namespace data{
struct gpu_header{
    uint32_t dimension_count;
    uint32_t column_count;
    uint32_t data_address_offset;
};

template<typename T = float>
inline structures::dynamic_struct<gpu_header, uint32_t> create_packed_header(const structures::data<T>& data, util::memory_view<const structures::buffer_info> buffers){
    assert(buffers.size() == data.columns.size());
    
    std::vector<VkDeviceAddress> device_addresses(buffers.size());
    for(int i: util::size_range(buffers))
        device_addresses[i] = util::vk::get_buffer_address(buffers[i]);
    
    structures::dynamic_struct<gpu_header, uint32_t> packed_header(10);

    return packed_header;
}
}
}