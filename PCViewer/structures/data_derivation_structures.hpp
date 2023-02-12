#pragma once
#include <enum_names.hpp>

namespace structures{
namespace data_derivation{
enum class execution: uint32_t{
    Cpu,
    Gpu,
    COUNT
};
static const enum_names<execution> execution_names{
    "Cpu",
    "Gpu"
};

struct workbench_settings{
    execution execution_backend{execution::Cpu};
};
}
}