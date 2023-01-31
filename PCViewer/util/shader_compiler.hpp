#pragma once
#include <string_view>
#include <robin_hood.h>
#include <vector>

namespace util{
namespace shader_compiler{
std::vector<uint32_t> compile(std::string_view code, const robin_hood::unordered_map<std::string, std::string>& defines = {});
}
}