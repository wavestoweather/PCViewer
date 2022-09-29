#pragma once
#include <string_view>
#include <descriptor_set_storage.hpp>

namespace util{
namespace global_descriptors{

static const std::string_view heatmap_descriptor_id{"heatmap"};

void setup_default_descriptors();
}
}