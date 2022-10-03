#pragma once
#include <attributes.hpp>
#include <change_tracker.hpp>
#include <data.hpp>
#include <half.hpp>
#include <vk_mem_alloc.h>
#include <vk_context.hpp>
#include <optional>

namespace structures{
template<typename T>
using const_unique = std::unique_ptr<const T>;

static const std::string_view default_templatelist_name{"On_load"};
struct templatelist{
    std::string                 name{};
    std::vector<uint32_t>       indices{};
    buffer_info                 gpu_indices{};
    std::vector<min_max<float>> min_maxs{};
    float                       point_ratio{};  // currently unused
    struct flags{
        bool identity_indices: 1;
    }                           flags{};
};

struct gpu_data_t{
    buffer_info                 header{};
    std::vector<buffer_info>    columns{};
};

struct dataset{
    std::string                         id{};
    std::string                         display_name{};
    std::string                         backing_data{};         // filename or folder of the data on the hdd
    size_t                              data_size{};
    std::vector<attribute>              attributes{};
    uint32_t                            original_attribute_size{};
    change_tracker<std::set<uint32_t>>  visible_attributes{};
    std::vector<const_unique<templatelist>> templatelists{};
    robin_hood::unordered_map<std::string_view, const templatelist*> templatelist_index;
    change_tracker<data<float>>         float_data{};
    change_tracker<data<half>>          half_data{};
    change_tracker<data<uint32_t>>      compressed_data{};
    gpu_data_t                          gpu_data{};
    struct data_flags{
        bool gpuStream: 1;          // data has to be streamed from ram to gpu as not enough space available
        bool cpuStream: 1;          // data has to be streamed from hdd to cpu as not enough space available
        bool half: 1;               // data is in half format
        bool cudaCompressed: 1;     // data is compressed
    }                                   data_flags{};

    // optional data for certain data types
    struct data_stream_infos{
        uint32_t                cur_block_index{};
        uint32_t                block_count{};
        size_t                  block_size{};
        bool                    forward_upload{};
    };
    std::optional<data_stream_infos>    gpu_stream_infos{};
    std::optional<data_stream_infos>    cpu_stream_infos{};

    bool operator==(const dataset& o) const {return id == o.id;}

    bool any_change() const {return visible_attributes.changed  || float_data.changed || half_data.changed || compressed_data.changed;}
    void clear_change()     {visible_attributes.changed = false; float_data.changed = false; half_data.changed = false; compressed_data.changed = false;}
};

}

namespace globals{

template<class T>
using changing = structures::change_tracker<T>;
using dataset_t = structures::unique_tracker<structures::dataset>;
using datasets_t = changing<std::map<std::string_view, dataset_t>>;
extern datasets_t datasets;

}