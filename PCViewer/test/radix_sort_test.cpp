#include <radix.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <set>
#include <robin_hood.h>
#include <stopwatch.hpp>
#include <ranges.hpp>
#include <string_view>
#include <radix_sort.hpp>
#include <numeric>
#include "test_commons.hpp"

#define BENCHMARK_STD_SORT
#ifdef BENCHMARK_STD_SORT
    #include <algorithm>
#endif

//#define BENCHMARKING_ENABLED

struct test_result{
    static const int success = 0;
    static const int transformer_error = 1;
    static const int order_fail = 2;
    static const int values_fail = 3;
};

template<class t>
int test_transformer(t input){
    radix::internal::transformer<t> tr{};
    if(tr(tr(input)) != input)
        return test_result::transformer_error;
    return test_result::success;
}

template<class t>
std::vector<t> get_random(size_t n, double max){
    std::vector<t> r(n);
    for(t& v: r)
        v = t(rand() / double(RAND_MAX) * max);
    return r;
}

template<class t>
int test_sort(size_t n, double max){
    auto rand_vals = get_random<t>(n, max);
    std::vector<t> tmp(n);

    robin_hood::unordered_map<t, size_t> value_histogram, sorted_histogram;
    for(t v: rand_vals)
        ++value_histogram[v];
    auto [res_begin, res_end] = radix::sort(rand_vals.begin(), rand_vals.end(), tmp.begin());

    // check increasing
    for(auto i = res_begin + 1; i != res_end; ++i){
        if(*(i - 1) > *i)
            return test_result::order_fail;
    }

    // check value histogram
    for(auto i = res_begin; i != res_end; ++i)
        ++sorted_histogram[*i];
    
    if(sorted_histogram != value_histogram)
        return test_result::values_fail;
    
    return test_result::success;
}

template<class t>
int test_sort_indirect(size_t n, double max){
    std::vector<t> rand_vals = get_random<t>(n, max);
    std::vector<uint32_t> index_sort(n);
    std::iota(index_sort.begin(), index_sort.end(), 0);

    robin_hood::unordered_map<uint32_t, size_t> value_histogram, sorted_histogram;
    for(auto v: index_sort)
        ++value_histogram[v];

    radix::sort_indirect(index_sort, [&](const uint32_t& v){return rand_vals[v];});

    // check increasing
    for(size_t i: util::i_range(size_t(1), n)){
        if(rand_vals[index_sort[i - 1]] > rand_vals[index_sort[i]])
            return test_result::values_fail;
    }

    // check histogram
    for(auto v: index_sort)
        ++sorted_histogram[v];
    
    if(sorted_histogram != value_histogram)
        return test_result::values_fail;

    return test_result::success;
}

template<class t>
int benchmark_radix_sort(size_t n, double max, std::string_view test_name, int iters = 10){
    std::vector<t> rand_vals(n), tmp(n);
    structures::stopwatch watch;
    watch.start();
    double avg_time{};
    for(int i: util::i_range(iters)){
        double diff = watch.lap();
        if(i > 0)
            avg_time += diff / iters;
        rand_vals = get_random<t>(n, max);
        watch.lap();
        radix::sort(rand_vals.begin(), rand_vals.end(), tmp.begin());
    }
    avg_time += watch.lap() / iters;
    std::cout << "[info] " << std::setw(15) << test_name << ": Sorting " << n << " numbers took on average " << avg_time << "ms" << std::endl;
    return test_result::success;
}

template<class t>
int benchmark_radix_sort_indirect(size_t n, double max, std::string_view test_name, int iters = 10){
    std::vector<t> rand_vals(n);
    std::vector<uint32_t> indices(n), tmp(n);
    std::iota(indices.begin(), indices.end(), 0);
    structures::stopwatch watch;
    watch.start();
    double avg_time{};
    for(int i: util::i_range(iters)){
        double diff = watch.lap();
        if(i > 0)
            avg_time += diff / iters;
        rand_vals = get_random<t>(n, max);
        watch.lap();
        radix::sort_indirect(indices.begin(), indices.end(), tmp.begin(), [&](uint32_t i){return rand_vals[i];});
    }
    avg_time += watch.lap() / iters;
    std::cout << "[info] " << std::setw(15) << test_name << ": Radix indirect for " << n << " numbers took on average " << avg_time << "ms" << std::endl;
    return test_result::success;
}

template<class t>
int benchmark_std_sort(size_t n, double max, std::string_view test_name, int iters = 10){
    std::vector<t> rand_vals(n);
    structures::stopwatch watch;
    watch.start();
    double avg_time{};
    for(int i: util::i_range(iters)){
        double diff = watch.lap();
        if(i > 0)
            avg_time += diff / iters;
        rand_vals = get_random<t>(n, max);
        watch.lap();
        std::sort(rand_vals.begin(), rand_vals.end());
    }
    avg_time += watch.lap() / iters;
    std::cout << "[info] " << std::setw(15) << test_name << ": std sorting " << n << " numbers took on average " << avg_time << "ms" << std::endl;
    return test_result::success;
}

int benchmark_inplace_radix(size_t n, double max, std::string_view test_name, int iters = 10){
    std::vector<uint8_t> rand_vals(n);
    std::vector<uint32_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    structures::stopwatch watch;
    watch.start();
    double avg_time{};
    for(int i: util::i_range(iters)){
        double diff = watch.lap();
        if(i > 0)
            avg_time += diff / iters;
        rand_vals = get_random<uint8_t>(n, max);
        watch.lap();
        radix::RadixSortMSDTransform(indices.data(), indices.size(), [&](uint32_t i){return rand_vals[i];}, 7);
    }
    avg_time += watch.lap() / iters;
    std::cout << "[info] " << std::setw(15) << test_name << ": Radix indirect inplace for " << n << " numbers took on average " << avg_time << "ms" << std::endl;
    return test_result::success;
}

int radix_sort_test(int, char** const){
    // transformer tests
    check_res(test_transformer(.5f));
    check_res(test_transformer(.5));
    check_res(test_transformer(uint8_t(5)));
    check_res(test_transformer(int8_t(-5)));
    check_res(test_transformer(uint16_t(5)));
    check_res(test_transformer(int16_t(-5)));
    check_res(test_transformer(uint32_t(5)));
    check_res(test_transformer(int32_t(-5)));
    check_res(test_transformer(uint64_t(5)));
    check_res(test_transformer(int64_t(-5)));

    // sorting tests
    check_res(test_sort<float>(100, 1.));
    check_res(test_sort<double>(100, 1.));
    check_res(test_sort<uint8_t>(100, 255.));
    check_res(test_sort<int8_t>(100, 100.));
    check_res(test_sort<uint16_t>(100, 5000.));
    check_res(test_sort<int16_t>(100, 5000.));
    check_res(test_sort<uint32_t>(100, 1000000.));
    check_res(test_sort<int32_t>(100, 1000000.));
    check_res(test_sort<uint64_t>(100, 10000000.));
    check_res(test_sort<int64_t>(100, 10000000.));

    check_res(test_sort_indirect<float>(100, 1.));
    check_res(test_sort_indirect<double>(100, 1.));
    check_res(test_sort_indirect<uint8_t>(100, 255.));
    check_res(test_sort_indirect<int8_t>(100, 100.));
    check_res(test_sort_indirect<uint16_t>(100, 5000.));
    check_res(test_sort_indirect<int16_t>(100, 5000.));
    check_res(test_sort_indirect<uint32_t>(100, 1000000.));
    check_res(test_sort_indirect<int32_t>(100, 1000000.));
    check_res(test_sort_indirect<uint64_t>(100, 10000000.));
    check_res(test_sort_indirect<int64_t>(100, 10000000.));

    // benchmarking
#ifdef BENCHMARKING_ENABLED
    constexpr int p = 20;
    check_res(benchmark_radix_sort<int64_t>(1 << p, 0, "int64 sorted"));
    check_res(benchmark_radix_sort<float>(1 << p, 1e10, "float"));
    check_res(benchmark_radix_sort<double>(1 << p, 1e10, "double"));
    check_res(benchmark_radix_sort<uint8_t>(1 << p, 255, "uint8"));
    check_res(benchmark_radix_sort<uint32_t>(1 << p, double(1U << 30), "uint32"));
    check_res(benchmark_radix_sort<uint64_t>(1 << p, double(1UL << 63), "uint64"));
    check_res(benchmark_radix_sort<int64_t>(1 << p, double(1UL << 63), "int64"));
#ifdef BENCHMARK_STD_SORT
    std::cout << std::endl;
    check_res(benchmark_std_sort<int64_t>(1 << p, 0, "int64 sorted"));
    check_res(benchmark_std_sort<float>(1 << p, 1e10, "float"));
    check_res(benchmark_std_sort<double>(1 << p, 1e10, "double"));
    check_res(benchmark_std_sort<uint8_t>(1 << p, 255, "uint8"));
    check_res(benchmark_std_sort<uint32_t>(1 << p, double(1U << 30), "uint32"));
    check_res(benchmark_std_sort<uint64_t>(1 << p, double(1UL << 63), "uint64"));
    check_res(benchmark_std_sort<int64_t>(1 << p, double(1UL << 63), "int64"));
#endif
    std::cout << std::endl;
    check_res(benchmark_radix_sort_indirect<int64_t>(1 << p, 0, "int64 sorted"));
    check_res(benchmark_radix_sort_indirect<float>(1 << p, 1e10, "float"));
    check_res(benchmark_radix_sort_indirect<double>(1 << p, 1e10, "double"));
    check_res(benchmark_radix_sort_indirect<uint8_t>(1 << p, 255, "uint8"));
    check_res(benchmark_radix_sort_indirect<uint32_t>(1 << p, double(1U << 30), "uint32"));
    check_res(benchmark_radix_sort_indirect<uint64_t>(1 << p, double(1UL << 63), "uint64"));
    check_res(benchmark_radix_sort_indirect<int64_t>(1 << p, double(1UL << 63), "int64"));
    std::cout << std::endl;
    check_res(benchmark_inplace_radix(1 << p, 255, "uint8"));
#endif

    std::cout << "[info] radix_sort_test successful" << std::endl;
    return test_result::success;
}