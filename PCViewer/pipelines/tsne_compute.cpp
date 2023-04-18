#include "tsne_compute.hpp"
#include <vk_initializers.hpp>
#include <vk_util.hpp>
#include <vma_initializers.hpp>
#include <util.hpp>
#include <complex>
#include <stager.hpp>
#include <thread>
#include "../gpu_radix_sort/radix_pipeline.hpp"

namespace tsne{namespace implementation{
inline float l2_squared(const float* a, const float* b, int dims){float r{}; for(int i: util::i_range(dims)) r += a[i] * b[i]; return r;}

std::pair<std::vector<float>, std::vector<int>> k_nearest_neighbors(util::memory_view<const float> points, int dims, int k){
    const int num_points = points.size() / dims;
    std::vector<std::thread> workers(std::thread::hardware_concurrency());
    std::atomic<int> cur_point{};
    std::vector<float> return_distances(num_points * k);
    std::vector<int>   return_indices(num_points * k);
    auto thread_func = [&](int thread_id){
        for(int i = cur_point++; cur_point < num_points; i = ++cur_point){
            std::vector<float> distances(k, std::numeric_limits<float>::max());
            std::vector<int>   indices(k);
            for(int j: util::i_range(num_points)){
                if(i == j)
                    continue;
                float d = l2_squared(points.data() + i * dims, points.data() + j * dims, dims);
                auto insert_pos = std::upper_bound(distances.begin(), distances.end(), d);
                auto dist = insert_pos - distances.begin();
                if(dist < k){
                    distances.insert(insert_pos, d); distances.pop_back();
                    indices.insert(indices.begin() + dist, j); indices.pop_back();
                }
            }
            assert(indices.size() == k && distances.size() == k);
            std::copy(distances.begin(), distances.end(), return_distances.begin() + i * k);
            std::copy(indices.begin(), indices.end(), return_indices.begin() + i * k);
        }
    };

    int t_id{};
    for(auto& w: workers)
        w = std::thread(thread_func, t_id++);
    
    for(auto& w: workers)
        w.join();
    return {std::move(return_distances), std::move(return_indices)};
}
}}

namespace pipelines{
tsne_compute::tsne_compute(){
    auto pool_info = util::vk::initializers::commandPoolCreateInfo(globals::vk_context.compute_queue_family_index);
    _command_pool = util::vk::create_command_pool(pool_info);
    auto fence_info = util::vk::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
    _fence = util::vk::create_fence(fence_info);

    // computation of denominator and y_tilde_spacings before startup as it will be always needed
    constexpr float h = 1.f / num_interpolation_points;
    std::array<float, num_interpolation_points> y_tilde_spacings;
    std::array<float, num_interpolation_points> denominator;
    for(int i: util::i_range(num_interpolation_points))
        y_tilde_spacings[i] = i * h + h / 2;
    for(int i: util::i_range(num_interpolation_points)){
        denominator[i] = 1;
        for(int j: util::i_range(num_interpolation_points))
            if(i != j)
                denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
    }
    // creating and uploading to the gpu
    auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, sizeof(y_tilde_spacings) + sizeof(denominator));
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    auto buffer = util::vk::create_buffer(buffer_info, alloc_info);
    std::array<float, num_interpolation_points * 2> upload_buffer;
    std::copy(y_tilde_spacings.begin(), y_tilde_spacings.end(), upload_buffer.begin());
    std::copy(denominator.begin(), denominator.end(), upload_buffer.begin() + y_tilde_spacings.size());
    structures::stager::staging_buffer_info upload_info{};
    upload_info.transfer_dir = structures::stager::transfer_direction::upload;
    upload_info.data_upload = util::memory_view(upload_buffer);
    upload_info.dst_buffer = buffer.buffer;
    globals::stager.add_staging_task(upload_info);

    _y_tilde_spacings = util::vk::get_buffer_address(buffer);
    _denominator = util::vk::get_buffer_address(buffer) + sizeof(y_tilde_spacings);

    globals::stager.wait_for_completion();  // waiting to ensure everything was uploaded
}

const tsne_compute::pipeline_data& tsne_compute::_get_or_create_pipeline(){
    if(!_pipeline_data.charges_qij_pipeline){
        auto shader_module = util::vk::create_scoped_shader_module(_charges_qij_path);
        auto stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        VkPushConstantRange push_constant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_charges_qij)};
        auto layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.charges_qij_layout = util::vk::create_pipeline_layout(layout_info);
        auto pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.charges_qij_layout, stage_create_info);
        _pipeline_data.charges_qij_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_copy_from_fft_output_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_copy_fft)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.copy_fft_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.copy_fft_layout, stage_create_info);
        _pipeline_data.copy_fft_from_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_copy_to_fft_input_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.copy_fft_layout, stage_create_info);
        _pipeline_data.copy_fft_to_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_copy_to_w_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_copy_to_w)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.copy_to_w_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.copy_to_w_layout, stage_create_info);
        _pipeline_data.copy_to_w_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_integration_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_integration)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.integration_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.integration_layout, stage_create_info);
        _pipeline_data.integration_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_interpolate_device_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_interpolate)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.interpolate_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.interpolate_layout, stage_create_info);
        _pipeline_data.interpolate_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_interpolate_indices_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_interpolate_indices)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.interpolate_indices_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.interpolate_indices_layout, stage_create_info);
        _pipeline_data.interpolate_indices_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_kernel_tilde_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_kernel_tilde)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.kernel_tilde_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.kernel_tilde_layout, stage_create_info);
        _pipeline_data.kernel_tilde_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_perplexity_search_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_perplexity_search)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.perplexity_search_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.perplexity_search_layout, stage_create_info);
        _pipeline_data.perplexity_search_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_qij_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij_qij)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_qij_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_qij_layout, stage_create_info);
        _pipeline_data.pij_qij_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_qij2_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij_qij2)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_qij2_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_qij2_layout, stage_create_info);
        _pipeline_data.pij_qij2_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_qij3_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij_qij3)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_qij3_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_qij3_layout, stage_create_info);
        _pipeline_data.pij_qij3_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_pij_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_pij)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.pij_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.pij_layout, stage_create_info);
        _pipeline_data.pij_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_point_box_index_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_point_box_index)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.point_box_index_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.point_box_index_layout, stage_create_info);
        _pipeline_data.point_box_index_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_potential_indices_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_potential_indices)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.potential_indices_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.potential_indices_layout, stage_create_info);
        _pipeline_data.potential_indices_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_repulsive_forces_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_repulsive_forces)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.repulsive_forces_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.repulsive_forces_layout, stage_create_info);
        _pipeline_data.repulsive_forces_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_sum_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_sum)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.sum_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.sum_layout, stage_create_info);
        _pipeline_data.sum_pipeline = util::vk::create_compute_pipeline(pipeline_info);

        shader_module = util::vk::create_scoped_shader_module(_upper_lower_bounds_path);
        stage_create_info = util::vk::initializers::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, *shader_module);
        push_constant = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_upper_lower_bounds)};
        layout_info = util::vk::initializers::pipelineLayoutCreateInfo({}, push_constant);
        _pipeline_data.upper_lower_bounds_layout = util::vk::create_pipeline_layout(layout_info);
        pipeline_info = util::vk::initializers::computePipelineCreateInfo(_pipeline_data.upper_lower_bounds_layout, stage_create_info);
        _pipeline_data.upper_lower_bounds_pipeline = util::vk::create_compute_pipeline(pipeline_info);
    }
    return _pipeline_data;
}

tsne_compute& tsne_compute::instance(){
    static tsne_compute singleton;
    return singleton;
}

tsne_compute::memory_info_t tsne_compute::compute_memory_size(const structures::tsne_options& options){
    constexpr int min_num_intervals = 125;
    constexpr int num_terms = 4;

    const size_t num_points = options.num_points;
    const int num_dims = options.num_dims;
    const int num_neighbors = options.num_neighbors;
    int       num_boxes_per_dim = min_num_intervals;

    std::array<int, 21> allowed_num_boxes_per_dim{25, 36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140, 150, 175, 200, 1125};
    if(num_boxes_per_dim < allowed_num_boxes_per_dim[20])
        num_boxes_per_dim = *std::lower_bound(allowed_num_boxes_per_dim.begin(), allowed_num_boxes_per_dim.end(), num_boxes_per_dim);
    
    const int num_total_boxes = num_boxes_per_dim * num_boxes_per_dim;
    const int num_total_interpolation_points = num_total_boxes * num_interpolation_points * num_interpolation_points;
    const int num_fft_coeffs_half = num_interpolation_points * num_boxes_per_dim;
    const int num_fft_coeffs = 2 * num_fft_coeffs_half;
    const int num_interpolation_points_1d = num_interpolation_points * num_boxes_per_dim;

    memory_info_t ret;
    size_t full_size{};
    ret.knn_indices_long = full_size; full_size += num_points * num_neighbors * sizeof(uint32_t);
    ret.pij_indices_device = full_size; full_size += num_points * num_neighbors * sizeof(int);
    ret.knn_squared_distances = full_size; full_size += num_points * num_neighbors * sizeof(float);
    ret.pij_non_symmetric = full_size; full_size += num_points * num_neighbors * sizeof(float);
    ret.pij = ret.knn_squared_distances;  // knn_squared_distances is not needed anymore, so bij takes over the storage
    ret.pij_workspace = ret.pij_non_symmetric; full_size += num_points * num_neighbors * sizeof(float); // pij_workspace takes over the workspace of pij_nonsymmetric and requires double the size of it
    ret.repulsive_forces = full_size; full_size += num_points * 2 * sizeof(float);
    ret.attractive_forces = full_size; full_size += num_points * 2 * sizeof(float);
    ret.gains = full_size; full_size += num_points * 2 * sizeof(float);
    ret.old_forces = full_size; full_size += num_points * 2 * sizeof(float);
    ret.normalization_vec = full_size; full_size += num_points * sizeof(float);
    ret.ones = full_size; full_size += num_points * 2 * sizeof(float); // vector for reduce summing, etc.
    // further resources in fit_tsn.cu line 274
    ret.point_box_idx = full_size; full_size += num_points * sizeof(int);
    ret.x_in_box = full_size; full_size += num_points * sizeof(float);
    ret.y_in_box = full_size; full_size += num_points * sizeof(float);
    ret.y_tilde_values = full_size; full_size += num_total_interpolation_points * num_terms * sizeof(float);
    ret.x_interpolated_values = full_size; full_size += num_points * num_interpolation_points * sizeof(float);
    ret.y_interpolated_values = full_size; full_size += num_points * num_interpolation_points * sizeof(float);
    ret.potentialsQij = full_size; full_size += num_points * num_terms * sizeof(float);
    ret.w_coefficients = full_size; full_size *= num_total_interpolation_points * num_terms + sizeof(float);
    ret.all_interpolated_values = full_size; full_size += num_terms * num_interpolation_points * num_interpolation_points * num_points * sizeof(float);
    ret.output_values = full_size; full_size += num_terms * num_interpolation_points * num_interpolation_points * num_points * sizeof(float);
    ret.all_interpolated_indices = full_size; full_size += num_terms * num_interpolation_points * num_interpolation_points * num_points * sizeof(int);
    ret.output_indices = full_size; full_size += num_terms * num_interpolation_points * num_interpolation_points * num_points * sizeof(int);
    ret.chargesQij = full_size; full_size += num_points * num_terms * sizeof(float);
    ret.box_lower_bounds = full_size; full_size += 2 * num_total_boxes * sizeof(float);
    ret.box_upper_bounds = full_size; full_size += 2 * num_total_boxes * sizeof(float);
    ret.kernel_tilde = full_size; full_size += num_fft_coeffs * num_fft_coeffs * sizeof(float);
    ret.fft_kernel_tilde = full_size; full_size += 2 * num_interpolation_points_1d * 2 * num_interpolation_points_1d * sizeof(std::complex<float>);
    ret.fft_input = full_size; full_size += num_terms * num_fft_coeffs + num_fft_coeffs * sizeof(float);
    ret.fft_w_coefficients = full_size; full_size += num_terms * num_fft_coeffs * (num_fft_coeffs_half + 1) * sizeof(std::complex<float>);
    ret.fft_output = full_size; full_size += num_terms * num_fft_coeffs * num_fft_coeffs * sizeof(float);
    ret.size = full_size;
    return ret;
}

void tsne_compute::record(VkCommandBuffer commands, const tsne_compute_info& info){
    

    const uint32_t dispatch_x = util::align(info.datapoint_count, _workgroup_size);
    const auto& pipeline_data = _get_or_create_pipeline();
    //vkCmdPushConstants(commands, pipeline_data.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(info), &info);
    //vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_data.pipeline);
    //vkCmdDispatch(commands, dispatch_x, 1, 1);
    // TODO missing pipelines


}

void tsne_compute::calculate(const tsne_calculate_info& info){
    // TODO move nearest neighbour calculation to gpu


    tsne_compute_info record_info{.dst_address = info.src_address, .src_address = info.dst_address, .tmp_address = info.tmp_address, .datapoint_count = as<uint32_t>(info.tsne_options.num_points), .dimension_count = as<uint32_t>(info.tsne_options.num_dims), .tmp_memory_infos = info.memory_info};

    // calculating
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()); util::check_vk_result(res);
    res = vkResetFences(globals::vk_context.device, 1, &_fence); util::check_vk_result(res);
    if(_command_buffer)
        vkFreeCommandBuffers(globals::vk_context.device, _command_pool, 1, &_command_buffer);
    _command_buffer = util::vk::create_begin_command_buffer(_command_pool);
    record(_command_buffer, record_info);
    util::vk::end_commit_command_buffer(_command_buffer, globals::vk_context.compute_queue.const_access().get(), {}, {}, {}, _fence);
}

void tsne_compute::calculate(const tsne_calculate_direct_info& info){
    // allocating temporary buffer
    auto memory_info = compute_memory_size(info.tsne_options);
    auto buffer_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, memory_info.size);
    auto alloc_info = util::vma::initializers::allocationCreateInfo();
    auto tmp_buffer = util::vk::create_buffer(buffer_info, alloc_info);

    tsne_calculate_info c_info{};
    c_info.dst_address = info.dst_address;
    c_info.src_address = info.src_address;
    c_info.tmp_address = util::vk::get_buffer_address(tmp_buffer);
    c_info.memory_info = memory_info;
    c_info.tsne_options = info.tsne_options;
    calculate(c_info);
}

void tsne_compute::calculate_cpu(const structures::tsne_options& info){
    auto src_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, info.num_points * info.num_dims * sizeof(float));
    auto src_buffer = util::vk::create_buffer(src_info, util::vma::initializers::allocationCreateInfo());
    structures::stager::staging_buffer_info upload_info{};
    upload_info.data_upload = util::memory_view(info.points, info.num_points * info.num_dims);
    upload_info.dst_buffer = src_buffer.buffer;
    globals::stager.add_staging_task(upload_info);
    
    auto dst_info = util::vk::initializers::bufferCreateInfo(VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, info.num_points * structures::tsne_options::num_reduced_dims * sizeof(float));
    auto dst_buffer = util::vk::create_buffer(dst_info, util::vma::initializers::allocationCreateInfo());

    tsne_calculate_direct_info c_info{};
    c_info.dst_address = util::vk::get_buffer_address(dst_buffer);
    c_info.src_address = util::vk::get_buffer_address(src_buffer);
    c_info.tsne_options = info;
    globals::stager.wait_for_completion();
    
    calculate(c_info);

    structures::stager::staging_buffer_info download_info{};
    download_info.transfer_dir = structures::stager::transfer_direction::download;
    download_info.data_download = util::memory_view(info.points, info.num_points * info.num_reduced_dims);
    download_info.dst_buffer = dst_buffer.buffer;
}

void tsne_compute::wait_for_fence(uint64_t timeout){
    auto res = vkWaitForFences(globals::vk_context.device, 1, &_fence, VK_TRUE, timeout);
    util::check_vk_result(res);
}
}