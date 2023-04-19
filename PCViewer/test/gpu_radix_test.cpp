#include "test_commons.hpp"
#include "vulkan_init.hpp"
#include <radix_pipeline.hpp>
#include <stager.hpp>

struct test_result{
    static const int success = 0;
    static const int flag_wrong = 1;
    static const int sorting_order_wrong = 2;
};

template<typename T, typename P = radix_sort::gpu::payload_none>
int test_radix_pipeline_creation(){
    auto& pipeline = radix_sort::gpu::radix_pipeline<T, P>::instance();
    //pipeline.record({}, {});
    return test_result::success;
}

template<typename T>
std::vector<T> get_randoms(int num_values);
template<> std::vector<int> get_randoms<int>(int num_values){
    std::vector<int> randoms(num_values);
    for(int i: util::i_range(num_values))
        randoms[i] = std::rand();
    return randoms;
}
template<> std::vector<uint32_t> get_randoms<uint32_t>(int num_values){
    std::vector<uint32_t> randoms(num_values);
    for(int i: util::i_range(num_values))
        randoms[i] = as<uint32_t>(std::rand());
    return randoms;
}
template<> std::vector<float> get_randoms<float>(int num_values){
    std::vector<float> randoms(num_values);
    for(int i: util::i_range(num_values))
        randoms[i] = as<float>(std::rand()) / RAND_MAX;
    return randoms;
}

template<typename T>
int test_radix_sort(int num_values = 1e3){
    auto& pipeline = radix_sort::gpu::radix_pipeline<T>::instance();
    std::vector<T> values(num_values);
    for(int i: util::i_range(num_values))
        values[i] = as<T>(i);
    auto original_values = values;

    std::random_shuffle(values.begin(), values.end());

    typename radix_sort::gpu::radix_pipeline<T>::sort_info_cpu sort_info{};
    sort_info.src_data = util::memory_view<const T>(values);
    sort_info.dst_data = util::memory_view<T>(values);
    pipeline.sort(sort_info);

    for(int i: util::i_range(num_values))
        if(original_values[i] != values[i])
            return test_result::sorting_order_wrong;
    
    return test_result::success;
}

int gpu_radix_test(int argc, char** const argv){
    vulkan_default_init();
    globals::stager.init();
    check_res(test_radix_pipeline_creation<float>());
    check_res(test_radix_pipeline_creation<int>());
    check_res(test_radix_pipeline_creation<uint32_t>());
    check_res((test_radix_pipeline_creation<float,radix_sort::gpu::payload_32bit>()));

    check_res(test_radix_sort<int>(100));

    std::cout << "[info] gpu_radix_test successful" << std::endl;
    globals::stager.cleanup();
    globals::vk_context.cleanup();
    return test_result::success;
}