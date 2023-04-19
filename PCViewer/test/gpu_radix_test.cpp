#include "test_commons.hpp"
#include "vulkan_init.hpp"
#include <radix_pipeline.hpp>

struct test_result{
    static const int success = 0;
    static const int flag_wrong = 1;
};

int test_radix_pipeline_creation(){
    auto& pipeline = radix_sort::gpu::radix_pipeline<float>::instance();
    //pipeline.record({}, {});
    return test_result::success;
}

int gpu_radix_test(int argc, char** const argv){
    vulkan_default_init();
    check_res(test_radix_pipeline_creation());

    std::cout << "[info] gpu_radix_test successful" << std::endl;
    globals::vk_context.cleanup();
    return test_result::success;
}