#pragma once

#include <brushes.hpp>
#include <string_view>
#include <buffer_info.hpp>
#include <enum_names.hpp>
#include <memory_view.hpp>

namespace pipelines{
class brusher{
    struct push_constants{
        VkDeviceAddress local_brush_address;
        VkDeviceAddress global_brush_address;
        VkDeviceAddress active_indices_address;
        VkDeviceAddress data_address;
        VkDeviceAddress index_buffer_address;
        uint32_t        local_global_brush_combine;
        uint32_t        data_size;
    };

    const std::string_view  compute_shader_path{"shader/brusher.comp.spv"};
    const uint32_t          shader_local_size{256};

    VkPipeline          _brushing_pipeline{};
    VkPipelineLayout    _brushing_pipeline_layout{};
    VkFence             _brush_fence{};
    VkCommandPool       _command_pool{};
    VkCommandBuffer     _command_buffer{};

    brusher();

public:
    enum struct brush_combination: uint32_t{
        and_c,
        or_c,
        COUNT
    };
    const structures::enum_names<brush_combination> brush_combination_names{
        "and",
        "or"
    };
    struct brush_info{
        std::string_view                drawlist_id{};
        brush_combination               brush_comb{};
        util::memory_view<VkSemaphore>  wait_semaphores{};
        util::memory_view<VkSemaphoreWaitFlags> wait_flags{};
        util::memory_view<VkSemaphore>  signal_semaphores{};
    };

    brusher(const brusher&) = delete;
    brusher& operator=(const brusher&) = delete;

    static brusher& instance();

    void brush(const brush_info& info);
};
}