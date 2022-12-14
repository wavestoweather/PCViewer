#include <radix.hpp>
#include <iostream>
#include <random>
#include <vector>

struct test_result{
    static const int success = 0;
    static const int order_fail = 1;
};

int radix_sort_test(int, char**){
    constexpr size_t n = 100;
    std::vector<float> vals(100), tmp;
    for(auto& v: vals)
        v = rand() / double(RAND_MAX);

    // sorting
    auto [res_begin, res_end] = radix::sort(vals.begin(), vals.end(), tmp.begin());
    // check
    for(auto i = res_begin + 1; i != res_end; ++i){
        if(*(i - 1) > *i){
            std::cout << "[radix_sort_test] failed sort check" << std::endl;
            return test_result::order_fail;
        }
    }

    return test_result::success;
}