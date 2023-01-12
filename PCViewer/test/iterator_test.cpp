#include "test_commons.hpp"
#include <ranges.hpp>
#include <vector>

struct test_result{
    static const int success = 0;
    static const int flag_wrong = 1;
};

int test_first_iterator(){
    std::vector<int> test(10);
    bool f = true;
    for(const auto& [v, first]: util::first_iter(test)){
        if(first != f)
            return test_result::flag_wrong;
        f = false;
    }
    return test_result::success;
}

int test_last_iterator(){
    std::vector<int> test(10);
    bool l = false;
    int c = 0;
    for(const auto& [v, last]: util::last_iter(test)){
        l = ++c == test.size();
        if(last != l)
            return test_result::flag_wrong;
    }
    return test_result::success;
}

int iterator_test(int argc, char** const argv){
    check_res(test_first_iterator());
    check_res(test_last_iterator());

    std::cout << "[info] iterator_test successful" << std::endl;
    return test_result::success;
}