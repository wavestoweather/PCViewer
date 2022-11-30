#pragma once
#include <buffer_info.hpp>
#include <drawlists.hpp>
#include <priority_globals.hpp>
#include <data_type.hpp>
#include <default_hash.hpp>
#include <gpu_sync.hpp>
#include <inttypes.h>

namespace structures{
namespace distance_calculator_structs{
struct pipeline_specs{
    data_type d_type{};
    
    bool operator==(const pipeline_specs& o) const {return d_type == o.d_type;}
};
struct pipeline_data{
    VkPipeline      pipeline;
    VkPipelineLayout pipeline_layout;
};
}
}

DEFAULT_HASH(structures::distance_calculator_structs::pipeline_specs);

namespace pipelines{
class distance_calculator{
    using pipeline_specs = structures::distance_calculator_structs::pipeline_specs;
    using pipeline_data = structures::distance_calculator_structs::pipeline_data;
    struct push_constants{
        VkDeviceAddress data_header_address{};
        VkDeviceAddress index_buffer_address{};
        VkDeviceAddress priority_distance_address{};
        uint32_t        data_size{};
        uint32_t        priority_attribute{};
        float           priority_center{};
        float           priority_distance{};
    };

    const std::string_view _compute_shader_path{"shader/distance_calculator.comp.spv"};

    // vulkan resources
    VkCommandPool                                           _command_pool{};
    VkCommandBuffer                                         _command_buffer{};
    VkFence                                                 _calculator_fence{};
    robin_hood::unordered_map<pipeline_specs, pipeline_data> _pipelines{};

    distance_calculator();

    const pipeline_data& _get_or_create_pipeline(const pipeline_specs& pipeline_specs);

public:
    struct distance_info{
        structures::data_type   data_type{};
        size_t                  data_size{};
        VkDeviceAddress         data_header_address{};
        VkDeviceAddress         index_buffer_address{};
        VkDeviceAddress         distances_address{};
        uint32_t                priority_attribute{};
        float                   priority_center{};
        float                   priority_distance{};

        structures::gpu_sync_info gpu_sync_info{};
    };

    distance_calculator(const distance_calculator&) = delete;
    distance_calculator& operator=(const distance_calculator&) = delete;

    static distance_calculator& instance();

    void calculate(const distance_info& info);
    void wait_for_fence(uint64_t timeout = std::numeric_limits<uint64_t>::max());
};
}