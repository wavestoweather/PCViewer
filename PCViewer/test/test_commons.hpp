#pragma once

#define check_res(x) if((x) != test_result::success){std::cout << "Failed for " << #x << " with error code " << (x) << std::endl; return (x);}
