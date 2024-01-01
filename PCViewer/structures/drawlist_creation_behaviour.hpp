#pragma once
#include <vector>
#include <string_view>

namespace structures{
// describes automatic actions done when a drawlist is created
struct drawlist_creation_behaviour{
    std::vector<std::string_view> coupled_workbenches;
};
}

namespace globals{
extern structures::drawlist_creation_behaviour drawlist_creation_behaviour;
}