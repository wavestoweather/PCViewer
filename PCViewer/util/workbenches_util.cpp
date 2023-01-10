#include "workbenches_util.hpp"

#include <data_workbench.hpp>
#include <parallel_coordinates_workbench.hpp>
#include <data_derivation_workbench.hpp>
#include <compression_workbench.hpp>
#include <scatterplot_workbench.hpp>
#include <images_workbench.hpp>
#include <load_behaviour.hpp>
#include <drawlist_creation_behaviour.hpp>
#include <drawlist_colors_workbench.hpp>
#include <violin_drawlist_workbench.hpp>
#include <violin_attribute_workbench.hpp>

namespace util{
namespace workbench{
void setup_default_workbenches(){
    const std::string_view data_wb_id{"Data workbench"};
    const std::string_view parallel_coordinates_wb_id{"Parallel coordinates workbench"};
    const std::string_view data_derivation_wb_id{"Data derivation workbench"};
    const std::string_view scatterplot_wb_id{"Scatterplot workbench"};
    const std::string_view compression_wb_id{"Compresssion workbench"};
    const std::string_view images_wb_id{"Images workbench"};
    const std::string_view violin_drawlist_wb_id{"Violin drawlist workbench"};
    const std::string_view violin_attribute_wb_id{"Violin attribute workbench"};

    // register all available workbenches -------------------------------------------
    auto data_wb = std::make_unique<workbenches::data_workbench>(data_wb_id);
    data_wb->active = true;
    globals::dataset_dependencies.push_back(data_wb.get());
    globals::primary_workbench = data_wb.get();
    globals::workbenches.emplace_back(std::move(data_wb));

    auto drawlist_color_wb = std::make_unique<workbenches::drawlist_colors_workbench>(globals::drawlist_color_wb_id);
    globals::workbenches.emplace_back(std::move(drawlist_color_wb));

    auto images_wb = std::make_unique<workbenches::images_workbench>(images_wb_id);
    globals::workbenches.emplace_back(std::move(images_wb));

    auto parallel_coordinates_wb = std::make_unique<workbenches::parallel_coordinates_workbench>(parallel_coordinates_wb_id);
    parallel_coordinates_wb->active = true;
    globals::dataset_dependencies.push_back(parallel_coordinates_wb.get());
    globals::drawlist_dataset_dependencies.push_back(parallel_coordinates_wb.get());
    globals::secondary_workbench = parallel_coordinates_wb.get();
    globals::workbenches.emplace_back(std::move(parallel_coordinates_wb));

    auto data_derivation_wb = std::make_unique<workbenches::data_derivation_workbench>(data_derivation_wb_id);
    globals::dataset_dependencies.push_back(data_derivation_wb.get());
    globals::workbenches.emplace_back(std::move(data_derivation_wb));

    auto scatterplot_wb = std::make_unique<workbenches::scatterplot_workbench>(scatterplot_wb_id);
    globals::dataset_dependencies.push_back(scatterplot_wb.get());
    globals::drawlist_dataset_dependencies.push_back(scatterplot_wb.get());
    globals::workbenches.emplace_back(std::move(scatterplot_wb));

    auto compression_wb = std::make_unique<workbenches::compression_workbench>(compression_wb_id);
    globals::workbenches.emplace_back(std::move(compression_wb));

    auto violin_drawlist_wb = std::make_unique<workbenches::violin_drawlist_workbench>(violin_drawlist_wb_id);
    globals::dataset_dependencies.push_back(violin_drawlist_wb.get());
    globals::drawlist_dataset_dependencies.push_back(violin_drawlist_wb.get());
    globals::workbenches.emplace_back(std::move(violin_drawlist_wb));
    
    auto violin_attribute_wb = std::make_unique<workbenches::violin_attribute_workbench>(violin_attribute_wb_id);
    globals::dataset_dependencies.push_back(violin_attribute_wb.get());
    globals::drawlist_dataset_dependencies.push_back(violin_attribute_wb.get());
    globals::workbenches.emplace_back(std::move(violin_attribute_wb));

    // load behavoiur setup ----------------------------------------------------------
    globals::load_behaviour.on_load.push_back({false, 1, {0, std::numeric_limits<size_t>::max()}});
    globals::drawlist_creation_behaviour.coupled_workbenches.push_back(parallel_coordinates_wb_id);
}
}
}