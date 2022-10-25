#pragma ocne
#include <memory_view.hpp>
#include <vk_context.hpp>
#include <robin_hood.h>
#include <gpu_sync.hpp>
#include <gpu_timing.hpp>

namespace structures{
namespace histogram_counter{
enum class data_type: uint32_t{
    float_type,
    half_type,
    uint_type,
    ushort_type
};
struct pipeline_specs{
    data_type d_type;

    bool operator==(const pipeline_specs& o) const {return d_type == o.d_type;}
};
struct pipeline_data{
    VkPipeline          pipeline;
    VkPipelineLayout    pipeline_layout;
};
}
}

template<> struct std::hash<structures::histogram_counter::pipeline_specs>{
    size_t operator()(const structures::histogram_counter::pipeline_specs& x) const{
        return util::memory_view<const uint32_t>(util::memory_view(x)).dataHash();
    }
};

namespace pipelines{

// can count up to 4d histogram
class histogram_counter{
    using pipeline_specs = structures::histogram_counter::pipeline_specs;
    using pipeline_data = structures::histogram_counter::pipeline_data;

    struct push_constants{
        VkDeviceAddress data_header_address{};
        VkDeviceAddress gpu_data_activations{};
        uint32_t a1{}, a2{}, a3{}, a4{};
        uint32_t s1{}, s2{}, s3{}, s4{};
        uint32_t data_size{};
    };

    const std::string_view _compute_shader_path{"shader/histogram_counter.comp.spv"};

    // vulkan resources
    structures::buffer_info                                 _count_info_buffer{};
    VkCommandPool                                           _command_pool{};
    VkCommandBuffer                                         _command_buffer{};
    VkFence                                                 _count_fence{};
    robin_hood::unordered_map<pipeline_specs, pipeline_data> _pipelines;

    // private constructor
    histogram_counter();

    const pipeline_data& _get_or_create_pipeline(const pipeline_specs& pipeline_specs);
    const structures::buffer_info& _get_or_resize_info_buffer(size_t byte_size);

public:
    struct count_info{
        structures::histogram_counter::data_type data_type{};
        size_t                              data_size{};
        VkDeviceAddress                     data_header_address{};
        VkDeviceAddress                     gpu_data_activations{};
        util::memory_view<uint32_t>         column_indices{};
        util::memory_view<uint32_t>         bin_sizes{};
        structures::gpu_sync_info           gpu_sync_info{};
        structures::gpu_timing_info*        gpu_timing_info{};
    };

    histogram_counter(const histogram_counter&) = delete;
    histogram_counter& operator=(const histogram_counter&) = delete;

    static histogram_counter& instance();

    void count(const count_info& info);
};
}