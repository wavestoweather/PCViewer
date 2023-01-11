#pragma ocne
#include <memory_view.hpp>
#include <vk_context.hpp>
#include <robin_hood.h>
#include <gpu_sync.hpp>
#include <gpu_timing.hpp>
#include <min_max.hpp>
#include <data_type.hpp>
#include <default_hash.hpp>

namespace structures{
namespace histogram_counter_structs{
enum class reduction_type_t: uint32_t{
    sum,
    min,
    max,
    COUNT
};
struct pipeline_specs{
    data_type_t     data_type{};
    uint32_t        dim_count{2};
    reduction_type_t reduction_type{reduction_type_t::sum};

    bool operator==(const pipeline_specs& o) const {return data_type == o.data_type && dim_count == o.dim_count && reduction_type == o.reduction_type;}
};
struct pipeline_data{
    VkPipeline          pipeline;
    VkPipelineLayout    pipeline_layout;
};
}
}

DEFAULT_HASH(structures::histogram_counter_structs::pipeline_specs);

namespace pipelines{
// can count up to 4d histogram
class histogram_counter{
    using pipeline_specs = structures::histogram_counter_structs::pipeline_specs;
    using pipeline_data = structures::histogram_counter_structs::pipeline_data;
    using reduction_type_t = structures::histogram_counter_structs::reduction_type_t;

    struct push_constants{
        VkDeviceAddress data_header_address{};
        VkDeviceAddress index_buffer_address{};
        VkDeviceAddress gpu_data_activations{};
        VkDeviceAddress histogram_buffer_address{};
        VkDeviceAddress priority_values_address{};
        uint32_t        a1{}, a2{}, a3{}, a4{};
        uint32_t        s1{}, s2{}, s3{}, s4{};
        float           a1_min{}, a2_min{}, a3_min{}, a4_min{};
        float           a1_max{}, a2_max{}, a3_max{}, a4_max{};
        uint32_t        data_size{};
    };

    const std::string_view _compute_shader_path{"shader/histogram_counter.comp.spv"};

    // vulkan resources
    structures::buffer_info                                 _count_info_buffer{};
    VkCommandPool                                           _command_pool{};
    VkCommandBuffer                                         _command_buffer{};
    VkFence                                                 _count_fence{};
    robin_hood::unordered_map<pipeline_specs, pipeline_data> _pipelines{};

    // private constructor
    histogram_counter();

    const pipeline_data& _get_or_create_pipeline(const pipeline_specs& pipeline_specs);
    const structures::buffer_info& _get_or_resize_info_buffer(size_t byte_size);

public:
    struct count_info{
        structures::data_type_t             data_type{};
        size_t                              data_size{};
        VkDeviceAddress                     data_header_address{};
        VkDeviceAddress                     index_buffer_address{};
        VkDeviceAddress                     gpu_data_activations{};
        structures::buffer_info             histogram_buffer{};
        VkDeviceAddress                     priority_values_address{};
        bool                                clear_counts{};
        util::memory_view<uint32_t>         column_indices{};
        util::memory_view<int>              bin_sizes{};
        util::memory_view<const structures::min_max<float>> column_min_max{};
        reduction_type_t                    reduction_type{reduction_type_t::sum};                                
        structures::gpu_sync_info           gpu_sync_info{};
        structures::gpu_timing_info*        gpu_timing_info{};
    };

    histogram_counter(const histogram_counter&) = delete;
    histogram_counter& operator=(const histogram_counter&) = delete;

    static histogram_counter& instance();

    void count(const count_info& info);
    void wait_for_fence(uint64_t timeout = std::numeric_limits<uint64_t>::max()); 
};
}