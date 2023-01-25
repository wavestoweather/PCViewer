#include "test_commons.hpp"
#include <kd_tree.hpp>
#include <random>
#include <stopwatch.hpp>

#define BENCHMARK

struct test_result{
    static const int success = 0;
    static const int error = 1;
};

int nearest_neighbour_test(){
    std::vector<std::vector<float>> datas(2, std::vector<float>(100));
    for(auto& v: datas)
        for(auto& f: v)
            f = rand() / double(RAND_MAX);
    structures::kd_tree::float_column_view column_view(2);
    uint32_t size{static_cast<uint32_t>(datas[0].size())};
    uint32_t dimensions{0};
    column_view[0].dimensionSizes = size;
    column_view[0].columnDimensionIndices = dimensions;
    column_view[0].cols = {deriveData::memory_view(datas[0])};
    column_view[1].dimensionSizes = size;
    column_view[1].columnDimensionIndices = dimensions;
    column_view[1].cols = {deriveData::memory_view(datas[1])};
    structures::kd_tree tree(column_view);
    for(size_t i: util::i_range(20)){
        size_t x = rand() % column_view.size();
        auto [neigh, dist] = tree.nearest_neighbour(x);
        for(size_t j: util::size_range(column_view)){
            float x_diff = column_view[0](0, j) - column_view[0](0, x);
            float y_diff = column_view[1](0, j) - column_view[1](0, x);
            if(x_diff * x_diff + y_diff * y_diff < dist)
                return test_result::error;
        }
    }
    return test_result::success;
}

int speed_benchmark(){
#ifdef BENCHMARK
    constexpr size_t s{1 << 30};
    std::vector<std::vector<float>> datas(2, std::vector<float>(s));
    for(auto& v: datas)
        for(auto& f: v)
            f = rand() / double(RAND_MAX);
    std::cout << "[info] Benchmark random numbers generated, starting timing" << std::endl;
    structures::stopwatch stopwatch; stopwatch.start();
    structures::kd_tree::float_column_view column_view(2);
    uint32_t size{static_cast<uint32_t>(datas[0].size())};
    uint32_t dimensions{0};
    column_view[0].dimensionSizes = size;
    column_view[0].columnDimensionIndices = dimensions;
    column_view[0].cols = {deriveData::memory_view(datas[0])};
    column_view[1].dimensionSizes = size;
    column_view[1].columnDimensionIndices = dimensions;
    column_view[1].cols = {deriveData::memory_view(datas[1])};
    structures::kd_tree tree(column_view);
    for(size_t i: util::i_range(20)){
        size_t x = rand() % column_view.size();
        auto [neigh, dist] = tree.nearest_neighbour(x);
        for(size_t j: util::size_range(column_view)){
            float x_diff = column_view[0](0, j) - column_view[0](0, x);
            float y_diff = column_view[1](0, j) - column_view[1](0, x);
            if(x_diff * x_diff + y_diff * y_diff < dist)
                return test_result::error;
        }
    }
    double ms = stopwatch.lap();
    std::cout << "[info] Needed " << ms << " ms for building the kd_tree for " << size << " datapoints and " << 20 << " random nearest neighbour queries" << std::endl;
#endif
    return test_result::success;
}

int kd_tree_test(int, char**const){
    check_res(nearest_neighbour_test());
    check_res(speed_benchmark());

    std::cout << "[info] Kd_tree_test successfull" << std::endl;
    return test_result::success;
}