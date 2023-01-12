#include "test_commons.hpp"
#include <ranges.hpp>

struct test_result{
    static const int success = 0;
    static const int error = 1;
};

int test_find(){
    std::vector<int> v{1, 2, 3, 5, 7, 9, 10};

    auto try_find_2 = v | util::try_find(2);
    auto try_find_6 = v | util::try_find(6);

    if(!try_find_2 || &v[1] != &try_find_2.value().get())
        return test_result::error;
    if(try_find_6)
        return test_result::error;
    try_find_2.value().get() = -1;
    return test_result::success;
}

int test_find_if(){
    std::vector<int> v{1, 2, 8, 3, 5, 7, 9, 10};

    auto try_find_four = v | util::try_find_if<int>([](const int& i){return i % 4 == 0;});
    auto try_find_2 = v | util::try_find_if<int>([](const int& i){return i == 2;});
    auto try_find_6 = v | util::try_find_if<int>([](const int& i){return i == 6;});
    
    if(!try_find_four || try_find_four.value() % 4 != 0)
        return test_result::error;
    if(!try_find_2 || &v[1] != &try_find_2.value().get())
        return test_result::error;
    if(try_find_6)
        return test_result::error;
    try_find_2.value().get() = -1;
    return test_result::success;
}

int ranges_test(int argc, char** const argv){
    check_res(test_find());
    check_res(test_find_if());

    std::cout << "[info] ranges_test successful" << std::endl;
    return test_result::success;
}