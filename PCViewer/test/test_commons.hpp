#pragma once

#include <iostream>
#define check_res(x) {int res = x; if((res) != test_result::success){std::cout << "Failed for " << #x << " with error code " << (res) << std::endl; return (res);} else{std::cout << "[info] " #x " successfull" << std::endl;}}
