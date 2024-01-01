#include "test_commons.hpp"
#include <shader_compiler.hpp>

struct test_result{
    static const int success = 0;
    static const int error = 1;
};

int test_basic_shader(){
    std::string shader_code = R"(
        #version 450

        layout(local_size_x = 32) in;
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };

        void main(){
            in_a[gl_GlobalInvocationID.x] = .0f;
        }
    )";
    try{
        auto res = util::shader_compiler::compile(shader_code);
    }
    catch(std::exception& e){
        std::cout << e.what() << std::endl;
        return test_result::error;
    }

    return test_result::success;
}

int shader_test(int, char**const){
    check_res(test_basic_shader());

    return test_result::success;
}