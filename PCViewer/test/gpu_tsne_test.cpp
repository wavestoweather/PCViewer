#include "test_commons.hpp"
#include "vulkan_init.hpp"
#include <tsne_compute.hpp>
#include <stager.hpp>

struct test_result{
    static const int success = 0;
    static const int flag_wrong = 1;
};

int test_pipeline_creation(){
    auto& pipeline = pipelines::tsne_compute::instance();
    //pipeline.record({}, {});
    return test_result::success;
}

int gpu_tsne_test(int argc, char** const argv){
    vulkan_default_init();
    globals::stager.init();
    check_res(test_pipeline_creation());

    std::cout << "[info] gpu_tsne_test successful" << std::endl;
    globals::stager.cleanup();
    globals::vk_context.cleanup();
    return test_result::success;
}