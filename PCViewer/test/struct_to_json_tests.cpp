#include "test_commons.hpp"
#include <parallel_coordinates_workbench.hpp>
#include <scatterplot_structs.hpp>

struct test_result{
    static const int success = 0;
    static const int parse_error = 1;
};

template<typename T>
bool check_struct(){
    // dump to string and parse
    T s;
    crude_json::value j = s;
    std::string j_string = j.dump();
    auto j_parsed = crude_json::value::parse(j_string);
    T s_parsed(j_parsed);
    // check for consistency
    if(s == s_parsed)
        return test_result::success;
    return test_result::parse_error;
}

int struct_to_json_tests(int argc, char** argv){
    check_res(check_struct<workbenches::parallel_coordinates_workbench::settings_t>());
    check_res(check_struct<structures::scatterplot_wb::settings_t>());

    return test_result::success;
}