#include "test_commons.hpp"
#include "vulkan_init.hpp"
#include <radix_pipeline.hpp>
#include <stager.hpp>
#include <random>
#include <algorithm>

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
int test_radix_sort(int num_values = 1e3, T offset = 0){
    auto& pipeline = radix_sort::gpu::radix_pipeline<T>::instance();
    pipeline.enable_gpu_timing_info();
    std::vector<T> values(num_values);
    for(int i: util::i_range(num_values))
        values[i] = as<T>(i) + offset;
    auto original_values = values;

    std::random_device d;
    std::mt19937 g(d());
    std::shuffle(values.begin(), values.end(), g);

    typename radix_sort::gpu::radix_pipeline<T>::sort_info_cpu sort_info{};
    sort_info.src_data = util::memory_view<const T>(values);
    sort_info.dst_data = util::memory_view<T>(values);
    auto begin = std::chrono::system_clock::now();
    pipeline.sort(sort_info);
    auto end = std::chrono::system_clock::now();
    std::cout << "[info] sorting took " << std::chrono::duration<double>(end - begin).count() << " s." << std::endl;

    for(int i: util::i_range(num_values))
        if(original_values[i] != values[i])
            return test_result::sorting_order_wrong;
    
    return test_result::success;
}

template<typename T, typename P>
int test_radix_sort_payload(int num_values = 1e3)
{
    std::vector<uint32_t> rotation(num_values);
    std::vector<T> keys(num_values);
    std::vector<P> payloads(num_values);
    for(int i: util::i_range(num_values)){
        rotation[i] = i;
        keys[i] = as<T>(i);
        payloads[i] = as<P>(i);
    }
    auto original_keys = keys;
    auto original_payloads = payloads;

    std::random_device d;
    std::mt19937 g(d());
    std::shuffle(rotation.begin(), rotation.end(), g);
    for(int i: util::i_range(num_values)){
        keys[i] = original_keys[rotation[i]];
        payloads[i] = original_payloads[rotation[i]];
    }

    auto begin = std::chrono::system_clock::now();
    if constexpr(sizeof(P) == sizeof(radix_sort::gpu::payload_32bit)){
        auto& pipeline = radix_sort::gpu::radix_pipeline<T, radix_sort::gpu::payload_32bit>::instance();
        pipeline.enable_gpu_timing_info();

        typename radix_sort::gpu::radix_pipeline<T, radix_sort::gpu::payload_32bit>::sort_info_cpu sort_info{};
        sort_info.src_data = util::memory_view<const T>(keys);
        sort_info.dst_data = util::memory_view<T>(keys);
        sort_info.payload_src_data = util::memory_view<const P>(payloads);
        sort_info.payload_dst_data = util::memory_view<P>(payloads);
        pipeline.sort(sort_info);
    }
    else if constexpr(sizeof(P) == sizeof(radix_sort::gpu::payload_64bit)){
        auto& pipeline = radix_sort::gpu::radix_pipeline<T, radix_sort::gpu::payload_64bit>::instance();
        pipeline.enable_gpu_timing_info();

        typename radix_sort::gpu::radix_pipeline<T, radix_sort::gpu::payload_64bit>::sort_info_cpu sort_info{};
        sort_info.src_data = util::memory_view<const T>(keys);
        sort_info.dst_data = util::memory_view<T>(keys);
        sort_info.payload_src_data = util::memory_view<const P>(payloads);
        sort_info.payload_dst_data = util::memory_view<P>(payloads);
        pipeline.sort(sort_info);
    }
    else
        assert(false && "unknown payload size");
    auto end = std::chrono::system_clock::now();
    std::cout << "[info] sorting took " << std::chrono::duration<double>(end - begin).count() << " s." << std::endl;

    for(int i: util::i_range(num_values))
        if(original_keys[i] != keys[i] || original_payloads[i] != payloads[i])
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

    // standard positive numbers
    //check_res(test_radix_sort<uint8_t>(255)); // not yet working
    //check_res(test_radix_sort<int8_t>(127));  // not yet working
    check_res(test_radix_sort<uint16_t>(1e4));
    check_res(test_radix_sort<int16_t>(1e4));
    check_res(test_radix_sort<uint32_t>(1e6));
    check_res(test_radix_sort<int>(1e6));
    check_res(test_radix_sort<float>(1e6));
    // testing with negative numbers
    check_res(test_radix_sort<int>(1e6, -40));
    check_res(test_radix_sort<float>(1e4, -1000.f));

    // testing with payload
    check_res((test_radix_sort_payload<uint32_t, uint32_t>(1e4)));
    check_res((test_radix_sort_payload<float, uint32_t>(1e4)));

    std::cout << "[info] gpu_radix_test successful" << std::endl;
    globals::stager.cleanup();
    globals::vk_context.cleanup();
    return test_result::success;
}