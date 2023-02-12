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
    uint32_t data_transform_offset;
};

template<typename T = float>
inline uint64_t header_size (const structures::data<T>& data) {
    uint64_t size = 4;                          // header fields
    size += data.dimension_sizes.size();        // dimension sizes
    size += data.column_dimensions.size();      // column dimension counts
    size += data.column_dimensions.size();      // colummn dimension offsets
    for(const auto& a: data.column_dimensions)
        size += a.size();                       // column dimension
    size *= sizeof(uint32_t);
    size += data.columns.size() * sizeof(uint64_t); // column data addresses
    size += 2 * data.column_transforms.size() * sizeof(uint32_t);// column transforms
    return size;
}

template<typename T>
inline structures::dynamic_struct<gpu_header, uint32_t> create_packed_header(const structures::data<T>& data, util::memory_view<const structures::buffer_info> buffers){
    assert(buffers.size() == data.columns.size());
    
    std::vector<VkDeviceAddress> device_addresses(buffers.size());
    for(size_t i: util::size_range(buffers))
        device_addresses[i] = util::vk::get_buffer_address(buffers[i]);
    
    size_t header_size = ::util::data::header_size(data);
    structures::dynamic_struct<gpu_header, uint32_t> packed_header((header_size - sizeof(gpu_header)) / sizeof(uint32_t));
    packed_header->dimension_count = static_cast<uint32_t>(data.dimension_sizes.size());
    packed_header->column_count = static_cast<uint32_t>(data.columns.size());

    // dimension sizes
    uint32_t cur_offset{};
    for(const auto& dim_size: data.dimension_sizes)
        packed_header[cur_offset++] = dim_size;
    
    // column dimension counts
    for(const auto& column_dimension: data.column_dimensions)
        packed_header[cur_offset++] = static_cast<uint32_t>(column_dimension.size());
    
    // reserving space for column dimension offsets
    uint32_t column_dimension_offset_index = cur_offset;
    cur_offset += static_cast<uint32_t>(data.column_dimensions.size());

    // column dimensions and their offset
    for(size_t i: size_range(data.column_dimensions)){
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

    // transform data
    if(data.column_transforms.size()){
        assert(data.column_transforms.size() == data.columns.size());
        packed_header->data_transform_offset = cur_offset;
        for(const auto& transform: data.column_transforms){
            packed_header[cur_offset++] = reinterpret_cast<const uint32_t&>(transform.scale);
            packed_header[cur_offset++] = reinterpret_cast<const uint32_t&>(transform.offset);
        }
    }

    return packed_header;
}

inline std::vector<uint32_t> active_attributes_to_indices(util::memory_view<const std::string_view> active_attributes, util::memory_view<const structures::attribute> dataset_attributes){
    std::vector<uint32_t> active_indices(active_attributes.size());
    for(auto&& [att, i]: util::enumerate(active_attributes)){
        auto at = att;
        active_indices[i] = static_cast<uint32_t>(dataset_attributes.index_of([at](const structures::attribute& a){return at == a.id;}));
    }
    return active_indices;
}

inline std::vector<uint32_t> active_attribute_refs_to_indices(util::memory_view<const std::reference_wrapper<const structures::attribute_info>> active_attributes, util::memory_view<const structures::attribute> dataset_attributes){
    std::vector<uint32_t> active_indices(active_attributes.size());
    for(auto&& [att, i]: util::enumerate(active_attributes)){
        auto at = att;
        active_indices[i] = static_cast<uint32_t>(dataset_attributes.index_of([at](const structures::attribute& a){return at.get().attribute_id == a.id;}));
    }
    return active_indices;
}

inline std::map<std::string_view, uint32_t> attribute_to_index(util::memory_view<const structures::attribute> attributes){
    std::map<std::string_view, uint32_t> ret;
    uint32_t i{0};
    for(const auto& att: attributes)
        ret[att.id] = i++;
    return ret;
}

inline uint32_t attribute_to_index_single(const std::string_view attribute, util::memory_view<const structures::attribute> attributes){
    return static_cast<uint32_t>(attributes.index_of([&attribute](const structures::attribute& a){return attribute == a.id;}));
}
}
}
