#include "workbenches_util.hpp"

#include <data_workbench.hpp>
#include <parallel_coordinates_workbench.hpp>
#include <data_derivation_workbench.hpp>
#include <load_behaviour.hpp>
#include <drawlist_creation_behaviour.hpp>

namespace util{
namespace workbench{
void setup_default_workbenches(){
    const std::string_view data_wb_id{"Data workbench"};
    const std::string_view parallel_coordinates_wb_id{"Parallel coordinates workbench"};
    const std::string_view data_derivation_wb_id{"Data derivation workbench"};

    // register all available workbenches
    auto data_wb = std::make_unique<workbenches::data_workbench>(data_wb_id);
    data_wb->active = true;
    globals::dataset_dependencies.push_back(data_wb.get());
    globals::primary_workbench = data_wb.get();
    globals::workbenches.emplace_back(std::move(data_wb));

    auto parallel_coordinates_wb = std::make_unique<workbenches::parallel_coordinates_workbench>(parallel_coordinates_wb_id);
    parallel_coordinates_wb->active = true;
    globals::dataset_dependencies.push_back(parallel_coordinates_wb.get());
    globals::drawlist_dataset_dependencies.push_back(parallel_coordinates_wb.get());
    globals::secondary_workbench = parallel_coordinates_wb.get();
    globals::workbenches.emplace_back(std::move(parallel_coordinates_wb));

    auto data_derivation_wb = std::make_unique<workbenches::data_derivation_workbench>(data_derivation_wb_id);
    globals::dataset_dependencies.push_back(data_derivation_wb.get());
    globals::workbenches.emplace_back(std::move(data_derivation_wb));

    globals::load_behaviour.on_load.push_back({false, 1, {0, std::numeric_limits<size_t>::max()}});
    globals::drawlist_creation_behaviour.coupled_workbenches.push_back(parallel_coordinates_wb_id);
}
}
}