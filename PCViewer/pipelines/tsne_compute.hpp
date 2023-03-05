#pragma once
#include <buffer_info.hpp>
#include <string_view>

namespace pipelines{
class tsne_compute{
    struct pipeline_data{
        VkPipeline       pipeline;
        VkPipelineLayout layout;
    };
    struct push_constants{
        VkDeviceAddress dst_address;
        VkDeviceAddress src_address;
        VkDeviceAddress tmp_address;
        uint32_t        datapoint_count;
        uint32_t        dimension_count;
    };
    static constexpr uint32_t     _workgroup_size = 1024;
    const std::string_view _compute_shader_path{"shader/distance_calculator.comp.spv"};
    const std::string_view _repulsive_forces_path{"shader/tsne_repulsive_forces.comp.spv"};
    const std::string_view _charges_qij_path{"shader/tsne_charges_qij.comp.spv"};
    const std::string_view _pij_path{"shader/tsne_pij.comp.spv"};
    const std::string_view _perplexity_search_path{"shader/tsne_pij.comp.spv"};
    const std::string_view _copy_to_fft_input_path{"shader/tsne_copy_to_fft.comp.spv"};
    const std::string_view _copy_from_fft_output_path{"shader/tsne_copy_from_fft.comp.spv"};
    const std::string_view _point_box_index_path{"shader/tsne_point_box_index.comp.spv"};
    const std::string_view _interpolate_device_path{"shader/tsne_interpolate_device.comp.spv"};
    const std::string_view _interpolate_indices_path{"shader/tsne_interpolate_indices.comp.spv"};
    const std::string_view _potential_indices_path{"shader/tsne_potential_indices.comp.spv"};
    const std::string_view _kernel_tilde_path{"shader/tsne_kernel_tilde.comp.spv"};
    const std::string_view _upper_lower_bounds_path{"shader/tsne_upper_lower_bounds.comp.spv"};
    const std::string_view _copy_to_w_path{"shader/tsne_copy_to_w.comp.spv"};
    const std::string_view _pij_qij_path{"shader/tsne_pij_qij.comp.spv"};
    const std::string_view _pij_qij2_path{"shader/tsne_pij_qij2.comp.spv"};
    const std::string_view _pij_qij3_path{"shader/tsne_pij_qij3.comp.spv"};
    const std::string_view _sum_path{"shader/tsne_sum.comp.spv"};
    const std::string_view _integration_path{"shader/tsne_integration.comp.spv"};

    // vulkan resources
    VkCommandPool       _command_pool{};
    VkCommandBuffer     _command_buffer{};
    VkFence             _fence{};
    pipeline_data       _pipeline_data{};

    tsne_compute();                                 // private constructor to forbid default construction

    const pipeline_data& _get_or_create_pipeline();  // might be changed to include pipeline specs
public:
    struct tsne_compute_info{
        VkDeviceAddress dst_address;
        VkDeviceAddress src_address;
        VkDeviceAddress tmp_address;
        uint32_t        datapoint_count;
        uint32_t        dimension_count;
    };

    tsne_compute(const tsne_compute&) = delete;
    tsne_compute& operator=(const tsne_compute&) = delete;

    static tsne_compute& instance();

    // direct pipeline recording to a command buffer
    void record(VkCommandBuffer commands, const tsne_compute_info& info);

    void calculate(const tsne_compute_info& info);
    void wait_for_fence(uint64_t timeout = std::numeric_limits<uint64_t>::max());
};
}