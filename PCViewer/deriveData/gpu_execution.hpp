#pragma once
#include <vulkan/vulkan.h>
#include <string_view>

#define workgroup_size 1024
#define STR_IND(s) STR(s)
#define STR(s) #s

namespace deriveData{
struct pipeline_info{
    VkPipeline          pipeline;
    VkPipelineLayout    layout;
    // maybe additional info
    std::vector<size_t> amt_of_threads; // multiple sizes used to be able to the reductions
    std::vector<std::vector<uint8_t>> push_constants_data;  // multiple push constants given for each called pipeline one

    // for more complex pipelines the prerecorded command buffer is put here
    VkCommandBuffer     recorded_commands;
};
using data_storage = std::variant<uint32_t, size_t, float>;

std::string optimize_operations(const std::string& input);

struct create_gpu_result{
    std::vector<structures::buffer_info> temp_gpu_buffers;  // used to hold additional gpu buffers such as reduction buffers
    std::vector<pipeline_info>           pipelines;
};
create_gpu_result create_gpu_pipelines(std::string_view instructions, VkCommandPool command_pool);
}