#include "test_commons.hpp"
#include "vulkan_init.hpp"
#include <radix_pipeline.hpp>

struct test_result{
    static const int success = 0;
    static const int flag_wrong = 1;
};

template<typename T, typename P = radix_sort::gpu::payload_none>
int test_radix_pipeline_creation(){
    auto& pipeline = radix_sort::gpu::radix_pipeline<T, P>::instance();
    //pipeline.record({}, {});
    return test_result::success;
}

int gpu_radix_test(int argc, char** const argv){
    vulkan_default_init();
    check_res(test_radix_pipeline_creation<float>());
    check_res(test_radix_pipeline_creation<int>());
    check_res(test_radix_pipeline_creation<uint32_t>());
    check_res((test_radix_pipeline_creation<float,radix_sort::gpu::payload_32bit>()));

    std::cout << "[info] gpu_radix_test successful" << std::endl;
    globals::vk_context.cleanup();
    return test_result::success;
}