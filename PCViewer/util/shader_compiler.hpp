#pragma once
#include <string>
#include <robin_hood.h>
#include <vector>
#include <string_view>

namespace util{
namespace shader_compiler{
std::vector<uint32_t> compile(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines = {});
std::vector<uint32_t> compile_hlsl(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines = {}, std::string_view entry_point = "main");
}
}