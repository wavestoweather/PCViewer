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
    uint32_t _;
};

template<typename T = float>
inline structures::dynamic_struct<gpu_header, uint32_t> create_packed_header(const structures::data<T>& data, util::memory_view<const structures::buffer_info> buffers){
    assert(buffers.size() == data.columns.size());
    
    std::vector<VkDeviceAddress> device_addresses(buffers.size());
    for(int i: util::size_range(buffers))
        device_addresses[i] = util::vk::get_buffer_address(buffers[i]);
    
    uint32_t header_size = data.header_size();
    structures::dynamic_struct<gpu_header, uint32_t> packed_header((header_size - sizeof(gpu_header)) / sizeof(uint));
    packed_header->dimension_count = data.dimension_sizes.size();
    packed_header->column_count = data.columns.size();

    // dimension sizes
    uint32_t cur_offset{};
    for(const auto& dim_size: data.dimension_sizes)
        packed_header[cur_offset++] = dim_size;
    
    // column dimension counts
    for(const auto& column_dimension: data.column_dimensions)
        packed_header[cur_offset++] = column_dimension.size();
    
    // reserving space for column dimension offsets
    uint32_t column_dimension_offset_index = cur_offset;
    cur_offset += data.column_dimensions.size();

    // column dimensions and their offset
    for(int i: size_range(data.column_dimensions)){
        packed_header[column_dimension_offset_index++] = cur_offset;
        for(auto j: data.column_dimensions[i])
            packed_header[cur_offset++] = j;
    }

    // data addresses
    packed_header->data_address_offset = cur_offset;
    for(auto addr: device_addresses){
        packed_header.reinterpret_at<uint64_t>(cur_offset) = addr;
        cur_offset += sizeof(addr) / sizeof(uint32_t);
    }

    return packed_header;
}
}
}