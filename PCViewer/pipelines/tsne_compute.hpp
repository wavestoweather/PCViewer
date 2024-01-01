#pragma once
#include <buffer_info.hpp>
#include <string_view>
#include <limits>
#include <robin_hood.h>
#include <tsne_options.hpp>

namespace tsne{namespace implementation{
//struct k_nearest_defaults{
//    // the follwing can be used to define a custom distance function
//    static float distance(const float* a, const float* b, int dims) {float s{}; for(int i: util::i_range(dims)) s += a[i] * b[i]; return std::sqrt(s);}
//};
// implements k nearest neighbour search with standard linear search (runs in O(n^2))
// a structure containing specialization functions can be provided
// for available overwrites see the k_nearest_default struct
//template<typename Params = k_nearest_defaults>
std::pair<std::vector<float>, std::vector<int>> static k_nearest_neighbors(util::memory_view<const float> points, int dimensions, int k);
}}
namespace pipelines{
class tsne_compute{
    struct pipeline_data{
        VkPipeline       charges_qij_pipeline;
        VkPipelineLayout charges_qij_layout;
        VkPipeline       copy_fft_from_pipeline;
        VkPipeline       copy_fft_to_pipeline;
        VkPipelineLayout copy_fft_layout;
        VkPipeline       copy_to_w_pipeline;
        VkPipelineLayout copy_to_w_layout;
        VkPipeline       integration_pipeline;
        VkPipelineLayout integration_layout;
        VkPipeline       interpolate_pipeline;
        VkPipelineLayout interpolate_layout;
        VkPipeline       interpolate_indices_pipeline;
        VkPipelineLayout interpolate_indices_layout;
        VkPipeline       kernel_tilde_pipeline;
        VkPipelineLayout kernel_tilde_layout;
        VkPipeline       perplexity_search_pipeline;
        VkPipelineLayout perplexity_search_layout;
        VkPipeline       pij_qij_pipeline;
        VkPipelineLayout pij_qij_layout;
        VkPipeline       pij_qij2_pipeline;
        VkPipelineLayout pij_qij2_layout;
        VkPipeline       pij_qij3_pipeline;
        VkPipelineLayout pij_qij3_layout;
        VkPipeline       pij_pipeline;
        VkPipelineLayout pij_layout;
        VkPipeline       point_box_index_pipeline;
        VkPipelineLayout point_box_index_layout;
        VkPipeline       potential_indices_pipeline;
        VkPipelineLayout potential_indices_layout;
        VkPipeline       repulsive_forces_pipeline;
        VkPipelineLayout repulsive_forces_layout;
        VkPipeline       sum_pipeline;
        VkPipelineLayout sum_layout;
        VkPipeline       upper_lower_bounds_pipeline;
        VkPipelineLayout upper_lower_bounds_layout;
    };
    struct pc_charges_qij{
        uint64_t chargesQij_address;
        uint64_t xs_address;
        uint64_t ys_address;
        uint32_t points_count;
        uint32_t n_terms;
    };
    struct pc_copy_fft{ // same push constants for copy_to/from_fft
        uint64_t fft_input_address;
        uint64_t w_coeffs_addresss;
        uint32_t fft_coeffs_count;
        uint32_t fft_coeffs_half_count;
        uint32_t terms_count;
    };
    struct pc_copy_to_w{
        uint64_t w_coefficients_address;
        uint64_t output_indices_address;
        uint64_t output_values_address;
        float elements_count;
    };
    struct pc_integration{
        uint64_t points_address;
        uint64_t attr_forces_address;
        uint64_t rep_forces_address;
        uint64_t gains_address;
        uint64_t old_forces_address;
        float eta;
        float normalization;
        float momentum;
        float exaggeration;
        uint32_t  points_count;
    };
    struct pc_interpolate{
        uint64_t interpolated_values_address;
        uint64_t y_in_box_address;
        uint64_t y_tilde_spacings_address;
        uint64_t denominator_address;
        uint32_t n_interpolation_points;
        uint32_t N;
    };
    struct pc_interpolate_indices{
        uint64_t w_coefficients_address;
        uint64_t point_box_indices_address;
        uint64_t chargesQij_address;
        uint64_t x_interpolated_values_address;
        uint64_t y_interpolated_values_address;
        uint32_t N;
        uint32_t n_interpolation_points;
        uint32_t n_boxes;
        uint32_t n_terms;
    };
    struct pc_kernel_tilde{
        uint64_t kernel_tilde_address;
        float x_min;
        float y_min;
        float h;
        uint32_t n_interpolation_points_1d;
        uint32_t n_fft_coeffs;
    };
    struct pc_perplexity_search{
        uint64_t betas_address;
        uint64_t lower_bound_address;
        uint64_t upper_bound_address;
        uint64_t found_address;
        uint64_t neg_entropy_address;
        uint64_t row_sum_address;
        float perplexity_target;
        float epsilon;
        uint32_t points_count;
    };
    struct pc_pij_qij{
        uint64_t points_address;
        uint64_t attr_forces_address;
        uint64_t pij_address;
        uint64_t coo_indices_address;
        uint32_t points_count;
        uint32_t nonzero_count;
    };
    struct pc_pij_qij2{
        uint64_t points_address;
        uint64_t attr_forces_address;
        uint64_t pij_address;
        uint64_t pij_row_address;
        uint64_t pij_col_address;
        uint32_t points_count;
    };
    struct pc_pij_qij3{
        uint64_t points_address;
        uint64_t pij_address;
        uint64_t pij_ind_address;
        uint64_t workspace_x_address;
        uint64_t workspace_y_address;
        uint32_t points_count;
        uint32_t neighbours_count;
    };
    struct pc_pij{
        uint64_t pij_address;
        uint64_t squared_dist_address;
        uint64_t betas_address;
        uint32_t points_count;
        uint32_t near_neighbours_count;
    };
    struct pc_point_box_index{
        uint64_t point_box_idx_address;
        uint64_t x_in_box_address;
        uint64_t y_in_box_address;
        uint64_t xs_address;
        uint64_t ys_address;
        uint64_t box_lower_bounds_address;
        float coord_min;
        float box_width;
        uint32_t n_boxes;
        uint32_t n_total_boxes;
        uint32_t N;
    };
    struct pc_potential_indices{
        uint64_t potentialsQij_address;
        uint64_t point_box_indices_address;
        uint64_t y_tilde_values_address;
        uint64_t x_interpolated_values_address;
        uint64_t y_interpolated_values_address;
        uint32_t N;
        uint32_t n_interpolation_points;
        uint32_t n_boxes;
        uint32_t n_terms;
    };
    struct pc_repulsive_forces{
        uint64_t repulsive_forces_address;
        uint64_t normalization_vec_address;
        uint64_t xs_address;
        uint64_t ys_address;
        uint64_t potentialsQij_address;
        uint32_t points_count;
        uint32_t n_terms;
    };
    struct pc_sum{
        uint64_t attr_forces_address;
        uint64_t workspace_x_address;
        uint64_t workspace_y_address;
        uint32_t points_count;
        uint32_t neighbours_count;
    };
    struct pc_upper_lower_bounds{
        uint64_t box_upper_bounds_address;
        uint64_t box_lower_bounds_address;
        float box_width;
        float x_min;
        float y_min;
        uint32_t n_boxes;
        uint32_t n_total_boxes;
    };
    static constexpr uint32_t _workgroup_size = 1024;
    const std::string_view _repulsive_forces_path{"shader/tsne_repulsive_forces.comp.spv"};
    const std::string_view _charges_qij_path{"shader/tsne_charges_qij.comp.spv"};
    const std::string_view _pij_path{"shader/tsne_pij.comp.spv"};
    const std::string_view _perplexity_search_path{"shader/tsne_perplexity_search.comp.spv"};
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

    // constant vulkan data which is calculated on init
    // the vulkan buffer info is not stored, as it is implicitly stored in the global context
    VkDeviceAddress     _y_tilde_spacings;
    VkDeviceAddress     _denominator;

    tsne_compute();                                 // private constructor to forbid default construction

    const pipeline_data& _get_or_create_pipeline();  // might be changed to include pipeline specs
public:
    struct memory_info_t{
        size_t
        // definitions in fit_tsne.cu line 160
        knn_indices_long,
        pij_indices_device,
        knn_squared_distances,
        pij_non_symmetric,
        pij,
        pij_workspace,
        repulsive_forces,
        attractive_forces,
        gains,
        old_forces,
        normalization_vec,
        ones,
        // further resources in fit_tsn.cu line 274
        point_box_idx,
        x_in_box,
        y_in_box,
        y_tilde_values,
        x_interpolated_values,
        y_interpolated_values,
        potentialsQij,
        w_coefficients,
        all_interpolated_values,
        output_values,
        all_interpolated_indices,
        output_indices,
        chargesQij,
        box_lower_bounds,
        box_upper_bounds,
        kernel_tilde,
        fft_kernel_tilde,
        fft_input,
        fft_w_coefficients,
        fft_output,
        size;
    };
    struct tsne_compute_info{
        VkDeviceAddress dst_address;
        VkDeviceAddress src_address;
        VkDeviceAddress tmp_address;
        uint32_t        datapoint_count;
        uint32_t        dimension_count;
        memory_info_t   tmp_memory_infos;
    };
    struct tsne_calculate_info{
        VkDeviceAddress          dst_address;
        VkDeviceAddress          src_address;
        VkDeviceAddress          tmp_address;
        memory_info_t            memory_info;
        structures::tsne_options tsne_options;
    };
    struct tsne_calculate_direct_info{
        VkDeviceAddress          dst_address;
        VkDeviceAddress          src_address;
        structures::tsne_options tsne_options;
    };
    struct tsne_calculate_cpu_info{
        structures::tsne_options tsne_options;
    };

    tsne_compute(const tsne_compute&) = delete;
    tsne_compute& operator=(const tsne_compute&) = delete;

    static tsne_compute& instance();

    static constexpr int num_interpolation_points = 3;

    // returns a map with the offsets for each needed storage buffer needed, to be able
    // to allocate a single buffer and use the offsets to get a distinct memory position
    // for each needed buffer
    // The "size" entry always holds the total size needed for the temporary buffer.
    memory_info_t compute_memory_size(const structures::tsne_options& options);

    // direct pipeline recording to a command buffer
    // all buffers have to be created in advance, contains only gpu code dispatch.
    // Cpu side implementations (iteration, ...) have to be done manually. See calculate() for
    // an example implementation
    void record(VkCommandBuffer commands, const tsne_compute_info& info);

    // src buffer has to contain the data, dst and temporary buffer have to be allocated with the correct sizes
    void calculate(const tsne_calculate_info& info);
    // src buffer has to contain the data, dst buffer has to be allocated with the correct size
    void calculate(const tsne_calculate_direct_info& info);
    // direct calculation from cpu buffers
    void calculate_cpu(const structures::tsne_options& info);

    void wait_for_fence(uint64_t timeout = std::numeric_limits<uint64_t>::max());
};
}