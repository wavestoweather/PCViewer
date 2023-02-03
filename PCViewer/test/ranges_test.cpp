#include "test_commons.hpp"
#include <ranges.hpp>
#include <change_tracker.hpp>
#include <map>
#include <string_view_util.hpp>

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

int test_rev_size_range(){
    std::vector<int> v(16);
    for(int i: util::rev_size_range(v))
        if(i < 0)
            return test_result::error;
    structures::change_tracker<std::vector<int>> c;
    c() = {16, 10, 13};
    for(int i: util::rev_size_range(c.read()))
        if(i < 0)
            return test_result::error;

    std::map<int, std::vector<int>> m{{2, {1,8,7}}};
    for(int i: util::rev_size_range(m[2]))
        if(i < 0)
            return test_result::error;
    return test_result::success;
}

int test_string_slice(){
    for(auto&& [slice, i]: util::enumerate("Hallo, das ist,, ein, schöner test" | util::slice(','))){
        std::cout << slice << std::endl;
        switch(i){
        case 0:
            if(slice != "Hallo")
                return test_result::error;
            break;
        case 1:
            if(slice != " das ist")
                return test_result::error;
            break;
        case 2:
            if(slice != "")
                return test_result::error;
            break;
        case 3:
            if(slice != " ein")
                return test_result::error;
            break;
        case 4:
            if(slice != " schöner test")
                return test_result::error;
            break;
        default:
            return test_result::error;
        }
    }
    for(auto&& [slice, i]: util::enumerate("Hallo| das ist,, ein| schöner test" | util::slice('|'))){
        std::cout << slice << std::endl;
        switch(i){
        case 0:
            if(slice != "Hallo")
                return test_result::error;
            break;
        case 1:
            if(slice != " das ist,, ein")
                return test_result::error;
            break;
        case 2:
            if(slice != " schöner test")
                return test_result::error;
            break;
        default:
            return test_result::error;
        }
    }
    return test_result::success;
}

int ranges_test(int argc, char** const argv){
    check_res(test_find());
    check_res(test_find_if());
    check_res(test_rev_size_range());
    check_res(test_string_slice());

    std::cout << "[info] ranges_test successful" << std::endl;
    return test_result::success;
}