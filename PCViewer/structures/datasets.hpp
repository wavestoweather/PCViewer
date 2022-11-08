#pragma once
#include <attributes.hpp>
#include <change_tracker.hpp>
#include <data.hpp>
#include <half.hpp>
#include <vk_mem_alloc.h>
#include <vk_context.hpp>
#include <optional>
#include <variant>

namespace structures{
template<typename T>
using const_unique = std::unique_ptr<const T>;

static const std::string_view templatelist_name_all_indices{"All indices"};
static const std::string_view templatelist_name_load_behaviour{"On load"};
struct templatelist{
    std::string                 name{};
    std::vector<uint32_t>       indices{};
    buffer_info                 gpu_indices{};
    std::vector<min_max<float>> min_maxs{};
    float                       point_ratio{};  // currently unused
    size_t                      data_size{};
    struct flags{
        bool identity_indices: 1;
    }                           flags{};
};

struct gpu_data_t{
    buffer_info                 header{};
    std::vector<buffer_info>    columns{};
};

using cpu_data_t = std::variant<data<float>, data<half>, data<uint32_t>>;

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
    change_tracker<cpu_data_t>          cpu_data{};
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
        size_t                  cur_block_size{};
        uint32_t                block_count{};
        size_t                  block_size{};
        bool                    forward_upload{};
    };
    std::optional<data_stream_infos>    gpu_stream_infos{};
    std::optional<gpu_data_t>           gpu_stream_data{};      // backing gpu buffer for async data loading
    std::optional<data_stream_infos>    cpu_stream_infos{};
    std::optional<cpu_data_t>           cpu_stream_data;        // backing buffer for async data loading

    bool operator==(const dataset& o) const {return id == o.id;}

    bool any_change() const {return visible_attributes.changed  || cpu_data.changed;}
    void clear_change()     {visible_attributes.changed = false; cpu_data.changed = false;}
};

}

namespace globals{

template<class T>
using changing = structures::change_tracker<T>;
using dataset_t = structures::unique_tracker<structures::dataset>;
using datasets_t = changing<std::map<std::string_view, dataset_t>>;
extern datasets_t datasets;
extern std::set<std::string_view> datasets_to_delete;   // is emptied in the main thread, only add delete tasks
}