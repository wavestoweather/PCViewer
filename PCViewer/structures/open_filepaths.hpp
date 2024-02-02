#pragma once
#include <string>
#include <vector>
#include <attributes.hpp>
#include <thread>

namespace globals{
extern std::thread file_dialog_thread;
extern std::vector<std::string> paths_to_open;
extern std::vector<structures::query_attribute> attribute_query;
}