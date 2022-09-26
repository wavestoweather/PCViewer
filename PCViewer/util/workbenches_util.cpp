#include "workbenches_util.hpp"

#include <data_workbench.hpp>
#include <parallel_coordinates_workbench.hpp>
#include <laod_behaviour.hpp>

namespace util{
namespace workbench{
void setup_default_workbenches(){
    // register all available workbenches
    auto data_wb = std::make_unique<workbenches::data_workbench>("Data workbench");
    data_wb->active = true;
    globals::dataset_dependencies.push_back(data_wb.get());
    globals::primary_workbench = data_wb.get();
    globals::workbenches.emplace_back(std::move(data_wb));

    auto parallel_coordinates_wb = std::make_unique<workbenches::parallel_coordinates_workbench>("Parallel coordinates workbench");
    parallel_coordinates_wb->active = true;
    globals::drawlist_dataset_dependencies.push_back(parallel_coordinates_wb.get());
    globals::secondary_workbench = parallel_coordinates_wb.get();
    globals::workbenches.emplace_back(std::move(parallel_coordinates_wb));

    globals::load_behaviour.on_load.push_back({false, 1, {0, std::numeric_limits<size_t>::max()}, {"Parallel coordinates workbench"}});
}
}
}